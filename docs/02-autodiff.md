# 02 — Autodiff

Motor mínimo de diferenciação automática, dois modos: reverso (`Tensor`,
estilo micrograd) e forward (`Dual`, números duais). Os dois são validados
de forma independente — um contra o outro, e ambos contra `torch.autograd`
e diferenças finitas centrais — porque nenhum autodiff deveria confiar só
em si mesmo pra provar que está certo.

---

## O motor reverso: três decisões de design

**Cada operação retorna um novo `Tensor` que carrega uma closure `_backward`.**
Quando `c = a * b`, o `Tensor` resultante guarda uma função que sabe
distribuir o gradiente de volta para `a` e `b` — a regra local da
multiplicação ($\partial c/\partial a = b$, $\partial c/\partial b = a$).
Isso é o que separa autodiff de diferenciação simbólica: nunca se constrói
uma expressão matemática explícita para a derivada, só um grafo de regras
locais que se aplicam em cascata.

**`backward()` percorre o grafo em ordem topológica reversa.** Constrói a
ordem por DFS (visita filhos antes de se adicionar à lista), depois
processa de trás pra frente. Isso garante que, quando `_backward()` de um
nó roda, todos os nós que dependem dele já processaram e já depositaram
sua contribuição de gradiente.

**Gradiente acumula com `+=`, nunca sobrescreve.** Se uma variável é usada
duas vezes no grafo (ex: $y = x \cdot x$), ela recebe contribuição de
gradiente por **dois caminhos diferentes**, e a regra da cadeia
multivariável diz que a derivada total é a **soma** das contribuições de
cada caminho. Sobrescrever em vez de acumular é o bug clássico que faz
autodiff parecer funcionar em exemplos simples e falhar silenciosamente
assim que uma variável é reutilizada.

Os três pontos foram validados contra `torch.autograd`: expressão de três
variáveis, variável reutilizada, e tensores vetoriais elementwise — os
quatro bateram exatamente.

## Composicionalidade: divisão sem regra própria

`__truediv__` é implementado como `self * other**-1` — não existe uma
`_backward` escrita à mão para divisão. O gradiente correto emerge
automaticamente da composição de duas regras já existentes (`mul` e
`pow`), porque o motor não sabe que está "dividindo": só está encadeando
multiplicação e potenciação, e a regra da cadeia cuida do resto.

Isso é o argumento central de por que autodiff escala para redes com
milhões de operações: escreve-se a regra local de um punhado de primitivas
(`+`, `*`, `pow`, `exp`), e qualquer composição delas — por mais profunda
que seja — deriva automaticamente, sem uma regra nova para cada combinação
possível.

Pela mesma razão, `x ** Tensor(...)` é bloqueado explicitamente: o motor
sabe derivar $x^n$ em relação a $x$ (regra do tombo), mas não em relação a
um expoente que também é variável — isso exigiria $\partial(x^y)/\partial y
= x^y \ln x$, uma primitiva que não foi implementada. É mais honesto barrar
com uma mensagem clara do que deixar alguém descobrir isso via um
gradiente silenciosamente errado.

---

## Diferenças finitas centrais: o oráculo dos oráculos

`numerical_gradient` serve como validação independente de qualquer outro
motor de autodiff — não depende do torch existir, só da definição de
derivada:

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(x + h e_i) - f(x - h e_i)}{2h}
$$

O erro de truncamento dessa aproximação é $O(h^2)$ — então "diminuir $h$"
parece sempre melhorar a precisão. Não melhora. Medido empiricamente numa
função com derivada conhecida em forma fechada ($f(x)=\sum x^3$):

| $h$ | erro relativo |
|---|---|
| $10^{-1}$ | $6.8\times10^{-3}$ |
| $10^{-3}$ | $6.8\times10^{-7}$ |
| $10^{-5}$ | $8.5\times10^{-11}$ ← ótimo |
| $10^{-8}$ | $9.8\times10^{-9}$ |
| $10^{-12}$ | $8.9\times10^{-5}$ |

O erro cai, atinge um mínimo perto de $10^{-5}$–$10^{-6}$, e sobe de novo
conforme $h$ continua diminuindo. Perto da precisão de máquina
($\varepsilon \approx 2.2\times10^{-16}$), a subtração
$f(x{+}h) - f(x{-}h)$ passa a ser dominada por ruído de arredondamento —
cancelamento catastrófico, a mesma doença numérica do Gram-Schmidt no
Módulo 1, só que aqui atingindo a própria definição de derivada, não um
efeito colateral de algoritmo. O ponto ótimo teórico fica perto de
$\varepsilon^{1/3}$ (não $\varepsilon^{1/2}$, que seria a intuição
ingênua) — vale revisitar a Fase 1.1 (erro em ponto flutuante) pra
entender por quê.

---

## Jacobiano, modo reverso: m passadas, uma por saída

Para $f: \mathbb{R}^n \to \mathbb{R}^m$, cada linha do Jacobiano vem de um
`backward()` independente, semeando a saída correspondente com gradiente
1 e as demais com 0. Isso custa **$m$ passadas completas** — reconstruindo
o grafo do zero a cada linha, porque este motor não retém o grafo entre
chamadas de backward (não há `retain_graph`, como no PyTorch). Sem reter,
nós internos compartilhados entre duas saídas acumulariam gradiente
incorretamente se o mesmo grafo fosse reusado sem zerar tudo — é
exatamente esse problema que o `retain_graph=True` do PyTorch resolve,
como opção explícita em vez de comportamento padrão.

O ponto estrutural: **modo reverso é ótimo quando $m \ll n$** — poucas
saídas, muitas entradas. É por isso que treinar redes neurais usa reverse
mode: a loss é escalar ($m=1$), os parâmetros são milhões ($n$ grande), e
um único `backward()` pega o gradiente inteiro.

---

## Modo forward: números duais

Um número dual $x + \varepsilon x'$, com $\varepsilon^2 = 0$, carrega
valor e derivada juntos, propagados num único passe pra frente. Não há
grafo, não há fase de backward — a derivada já sai pronta na parte dual
no mesmo passe que calcula o valor. A regra do produto sai de graça da
álgebra: $(a+\varepsilon a')(b+\varepsilon b') = ab + \varepsilon(ab'+a'b)$,
já que $\varepsilon^2$ descarta o termo cruzado.

`jacobian_forward` é o espelho exato do reverso: em vez de $m$ passadas
(cada uma dando uma linha inteira), são **$n$ passadas** (cada uma dando
uma coluna inteira — todas as saídas, para uma direção de entrada).

### A prova, com contagem exata de chamadas

Duas funções, direções opostas de $n$ vs. $m$, instrumentadas pra contar
quantas vezes `f` é avaliada:

| Caso | $n$ (entradas) | $m$ (saídas) | chamadas forward | chamadas reverso |
|---|---|---|---|---|
| "largo" | 6 | 1 | **6** | **1** |
| "alto" | 2 | 5 | **2** | **5** |

Os dois modos concordam exatamente no resultado (validado por
`test_forward_and_reverse_modes_agree`) — a única coisa que muda é o
custo, e o custo segue $n$ para forward, $m$ para reverso, sem exceção.
**Não existe modo universalmente melhor**: a escolha depende inteiramente
da forma de $f$. Redes neurais têm $m=1$ (a loss) e $n$ em milhões —
reverso vence sempre, por uma margem gigantesca. Um sistema de equações
com poucas entradas e muitas saídas inverteria a escolha.

---

## Fechando o argumento do módulo

Cinco peças — `Tensor`, composicionalidade, diferenças finitas,
Jacobiano reverso, números duais — convergem para um só ponto: **o modo
"certo" de autodiff não existe isolado da forma do problema.** Reverso
ganha quando a saída é escalar e as entradas são muitas (o caso de
treinar redes); forward ganha no caso oposto. E diferenças finitas — que
não é nem forward nem reverso, é só a definição de derivada aplicada
ingenuamente — continua sendo indispensável não porque seja rápido, mas
porque é o único dos três que não pode estar errado da mesma forma que os
outros dois: qualquer bug de regra de `_backward` ou de `Dual` se
denuncia contra ele.
