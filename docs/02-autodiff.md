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

---

## Hessiana: um híbrido deliberado, não autodiff de segunda ordem

Este motor não faz diferenciação automática de segunda ordem "de verdade".
Isso exigiria que a própria passada de `backward()` fosse parte de um
grafo diferenciável — cada operação dentro de `_backward()` teria que ser
construída com `Tensor`s, não com aritmética crua de `numpy`, para que um
segundo `backward()` pudesse propagar através da primeira passada. É
exatamente o que o PyTorch faz com `create_graph=True`, e é uma extensão
real do motor, não um ajuste pequeno — cada uma das clousures de
`_backward` teria que virar, ela mesma, uma composição de `Tensor`s
diferenciável.

Em vez disso, `hessian()` é um híbrido deliberado:

$$
H_{:,j} \approx \frac{\nabla f(x_0 + h\,e_j) - \nabla f(x_0 - h\,e_j)}{2h}
$$

O gradiente $\nabla f$ vem de `gradient()` — **exato**, via `Tensor.backward()`,
sem nenhum erro de truncamento. A diferença finita central é aplicada
**sobre esse gradiente já exato**, não sobre $f$ diretamente. O resultado:
metade da Hessiana vem de autodiff (a parte que cada coluna representa —
o vetor gradiente inteiro, em cada ponto perturbado), e a outra metade
vem de diferença finita (a forma como as colunas se combinam). É mais
barato que autodiff de segunda ordem completo, e mais preciso que aplicar
diferença finita duas vezes em cascata (o que dobraria o erro de
truncamento $O(h^2)$ acumulado).

## O teste que teria pego um bug real: simetria de Schwarz

`test_hessian_is_symmetric` não é uma checagem incidental — ele codifica o
**teorema de Schwarz** (as derivadas parciais mistas comutam,
$\partial^2f/\partial x_i\partial x_j = \partial^2f/\partial x_j\partial x_i$,
para $f$ suficientemente suave). Numa implementação de Hessiana via
double-backward de verdade, essa simetria sairia garantida pela própria
estrutura do grafo computacional. Aqui, **não é garantida por construção**
— `hessian()` calcula cada coluna de forma independente, perturbando uma
variável de cada vez, sem nenhuma lógica que force $H_{ij} = H_{ji}$
explicitamente.

A simetria observada (~$10^{-11}$ de assimetria residual, só ruído de
ponto flutuante) é uma propriedade **emergente** de dois processos
numéricos independentes — gradiente exato mais diferença finita —
concordando porque a matemática subjacente é simétrica, não porque o
código garante. Se esse teste falhasse um dia, seria sinal de bug real na
implementação (ex: um erro de índice trocando $i$ e $j$ em algum lugar),
não de imprecisão numérica esperada — é exatamente o tipo de teste que
vale manter mesmo depois que "parece óbvio que vai passar".

## Fechando o cálculo vetorial da Fase 1.1

Gradiente, Jacobiano, Hessiana — as três derivadas que a Fase 1.1 pedia
("gradientes, Jacobianos, Hessianos, regra da cadeia") agora têm
implementação própria, validada por três oráculos independentes que nunca
concordam por acidente: `torch.autograd` (outro motor de autodiff inteiro),
diferenças finitas centrais (a definição de derivada, sem nenhum motor),
e — no caso da Hessiana — o teorema de Schwarz (uma propriedade
matemática que o código não impõe, só herda).

---

## O capstone: regressão logística, do zero, contra o sklearn

Última peça do motor: `sum()`, a redução que faltava pra fazer produto
escalar entre um vetor de pesos e um vetor de features —
$z = \sum_i w_i x_i + b$ — sem ela, não havia como colapsar oito
contribuições numa saída escalar única. Como toda peça deste módulo, o
gradiente de `sum()` é simples e mecânico: $\partial(\sum_i x_i)/\partial
x_i = 1$ para todo $i$, o gradiente de saída se espalha igual de volta
pra cada elemento que entrou na soma.

Com `sum()`, `log()`, `exp()` e `SGD`, a regressão logística inteira —
sigmoide, perda de entropia cruzada, laço de treino — é construída por
composição, sem nenhuma primitiva nova. `sigmoid(z) = 1/(1+exp(-z))` e a
perda usam só o que já existia. Esse é o argumento de composicionalidade
do módulo levado até o fim: seis primitivas (`+`, `*`, `pow`, `exp`,
`log`, `sum`) bastam pra treinar um classificador de verdade.

### O teste que prova que o motor funciona, não só que compila

Treinado nos dados HVI reais do Módulo 0 — 180 fardos de treino, rótulo
sintético "premium" via combinação linear de resistência, uniformidade,
comprimento e impureza mais ruído gaussiano (um limiar linear com
sobreposição de classes real, não um problema trivialmente separável) —
o modelo bateu **exatamente** a acurácia de teste do
`sklearn.linear_model.LogisticRegression` no mesmo split (0.7143 nos
dois), e os pesos aprendidos ficaram próximos coeficiente a coeficiente:

| Feature | Nosso peso | Peso do sklearn |
|---|---|---|
| strength | 0.779 | 0.895 |
| uniformity | 0.647 | 0.699 |
| uhml | 0.633 | 0.608 |
| trash | -0.665 | -0.752 |

As pequenas diferenças vêm do otimizador: nosso SGD puro, full-batch, 120
passos; o `sklearn` usa L-BFGS com regularização L2 leve por padrão — dois
caminhos de otimização diferentes convergindo pra perto do mesmo mínimo,
porque o problema é bem-condicionado o suficiente pra isso importar pouco.

Isso fecha o argumento central do módulo inteiro: um motor de ~200 linhas,
validado peça por peça contra `torch`, diferenças finitas centrais, e
propriedades matemáticas (Schwarz, teorema de Rayleigh), compõe um sistema
que treina um modelo real e generaliza tão bem quanto uma biblioteca de
produção — não por coincidência, mas porque cada peça foi provada correta
antes de compor a próxima.
