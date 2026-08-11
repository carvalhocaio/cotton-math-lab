# 04 — Otimização

Seis otimizadores de primeira ordem, mesma interface (`step()`/`zero_grad()`
do `SGD` do Módulo 2), cada um validado passo a passo contra o
`torch.optim` equivalente — não só "converge", trajetória idêntica. Depois,
Newton e BFGS, reaproveitando `gradient()` e `hessian()` do Módulo 2 sem
nenhuma matemática nova. O fio condutor: cada método troca uma dor por
outra, e a dor certa depende da geometria do problema, não de qual método
está na moda.

---

## Momentum: acumula direção, mas amplia o passo efetivo

$$
v \leftarrow \text{momentum} \cdot v + \nabla\theta, \qquad \theta \leftarrow \theta - \text{lr} \cdot v
$$

Validado contra `torch.optim.SGD(momentum=...)`, trajetória idêntica.

### O achado que não seguiu o roteiro esperado

Primeira tentativa, mesmo `lr` para SGD e Momentum na superfície
$f(x,y)=x^2+10y^2$: Momentum ficou **786× pior** que SGD puro. Não é bug —
Momentum amplia o passo efetivo por um fator perto de $1/(1-\text{momentum})$
(com momentum=0.9, isso é 10×), e esse fator pode desestabilizar a direção
de maior curvatura mesmo quando o `lr` nominal parecia razoável. Variando
`lr`:

| lr | SGD | Momentum | razão |
|---|---|---|---|
| 0.005 | 4.05 | 1.38 | 0.34 |
| 0.02 | 0.344 | **0.012** | **0.034** (29× melhor) |
| 0.03 | 0.064 | 0.118 | 1.86 (Momentum já pior) |

A vantagem existe, e é grande — mas numa janela estreita de `lr`, não
universalmente. "Momentum acelera convergência" é verdade condicional, não
incondicional.

---

## Nesterov: a mesma ideia, olhando à frente

$$
v \leftarrow \text{momentum}\cdot v + \nabla\theta, \qquad \theta \leftarrow \theta - \text{lr}\cdot(\nabla\theta + \text{momentum}\cdot v)
$$

O PyTorch (e esta implementação) usa uma reformulação algébrica que evita
uma segunda avaliação forward numa posição futura — mesmo resultado, sem o
custo extra. Validado contra `torch.optim.SGD(nesterov=True)`, diferença
$6.8\times10^{-8}$.

### Resolve a instabilidade do ciclo anterior — de verdade

No mesmo `lr=0.03` onde Momentum clássico ficou pior que SGD (0.118 vs
0.064), Nesterov chegou a **0.0103** — melhor que os dois. Em `lr=0.04`,
Momentum piora ainda mais (0.768); Nesterov continua estável (0.0007). A
correção de "olhar à frente" desloca a fronteira de estabilidade — não é
só elegância teórica.

---

## RMSProp: muda de eixo — escala, não direção

$$
v \leftarrow \alpha v + (1-\alpha)(\nabla\theta)^2, \qquad \theta \leftarrow \theta - \text{lr}\cdot\frac{\nabla\theta}{\sqrt{v}+\epsilon}
$$

Validado contra `torch.optim.RMSprop`, diferença $\sim10^{-8}$.

### A prova de que ele equaliza escalas, com um número

Na superfície com curvatura 10× maior em $y$ que em $x$: com SGD, o passo
em $y$ foi **6.8× maior** que em $x$ — a curvatura vira direto passo
desigual. Com RMSProp, a razão caiu pra **1.000**. Os dois parâmetros
avançam igual, apesar da curvatura brutalmente diferente — a adaptação por
parâmetro funcionando, não uma alegação de manual.

Consequência prática: RMSProp converge bem numa faixa de `lr` **muito**
mais ampla que Momentum precisou (testado 0.1 a 0.3, todos convergindo a
perda $<10^{-4}$) — não precisa caçar o `lr` certo do mesmo jeito.

---

## Adam: os dois eixos juntos, mais correção de viés

$$
m \leftarrow \beta_1 m + (1-\beta_1)\nabla\theta, \quad v \leftarrow \beta_2 v + (1-\beta_2)(\nabla\theta)^2
$$
$$
\hat{m} = \frac{m}{1-\beta_1^t}, \quad \hat{v} = \frac{v}{1-\beta_2^t}, \quad \theta \leftarrow \theta - \text{lr}\cdot\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon}
$$

Validado contra `torch.optim.Adam`, diferença $\sim10^{-7}$.

### Por que a correção de viés existe: um invariante exato

Em $t=1$, a correção cancela exatamente o fator de viés introduzido:
$\hat{m}=\nabla\theta$ e $\hat{v}=(\nabla\theta)^2$, então
$\hat{m}/\sqrt{\hat{v}} = \text{sign}(\nabla\theta)$ — **o primeiro passo do
Adam é sempre $\pm\text{lr}$, independente da magnitude do gradiente
inicial**. Testado com gradientes de 0.001 a 500: todos deram passo
$\approx\text{lr}$, até a quarta casa decimal.

Sem a correção, o primeiro passo fica inflado por um fator
$(1-\beta_1)/\sqrt{1-\beta_2}$ — com os betas padrão, $\approx3.16\times$
maior que deveria. A raiz só entra do lado de $v$ (que carrega
$(\nabla\theta)^2$); o lado de $m$ não tem raiz — essa assimetria entre
como $\beta_1$ e $\beta_2$ enviesam suas respectivas médias é exatamente o
que a correção resolve.

---

## AdamW: por que não é "Adam + weight decay"

$$
\theta \leftarrow \theta - \text{lr}\cdot\left(\frac{\hat{m}}{\sqrt{\hat{v}}+\epsilon} + \text{weight\_decay}\cdot\theta\right)
$$

Validado contra `torch.optim.AdamW`, diferença $\sim10^{-6}$.

### A demonstração com dois parâmetros de histórico diferente

Dois parâmetros, mesmo valor inicial, históricos de gradiente radicalmente
diferentes (um recebe gradiente 10.000× maior que o outro por 20 passos),
depois gradiente real zerado — só o efeito do `weight_decay` observado:

| Método | decaimento relativo (histórico grande) | decaimento relativo (histórico pequeno) |
|---|---|---|
| AdamW | 56.93% | 56.93% |
| "Adam + L2 no gradiente" | 43.6% | 68.6% |

Com AdamW, os dois decaem **na mesma proporção** — o `weight_decay`
nominal é a força de regularização real, ponto final. Com a forma ingênua
(somar `weight_decay·θ` ao gradiente antes do Adam processar), o termo de
decaimento entra nas médias móveis e fica reescalado pela adaptação por
parâmetro: o parâmetro com $v$ grande (histórico de gradiente grande)
recebe **menos** decaimento efetivo; o de $v$ pequeno recebe **mais**. A
força de regularização passa a depender do histórico de treino de cada
parâmetro — não é o que ninguém pretende quando escolhe um
`weight_decay=0.01`.

`AdamW` herda de `Adam` por código (só `step()` muda) — a única vez no
módulo em que um otimizador reaproveita outro por herança, porque aqui a
relação genuinamente é "a mesma coisa, mais um termo", diferente de
Momentum→Nesterov ou RMSProp→Adam, que são famílias de regra distintas.

---

## Rosenbrock: o benchmark que desmente o clichê "Adam é sempre melhor"

$$
f(x,y) = (1-x)^2 + 100(y-x^2)^2, \quad \text{mínimo em } (1,1)
$$

Vale estreito e **curvo** (segue a parábola $y=x^2$) — diferente da
quadrática dos ciclos anteriores, cujos eixos principais coincidem com
$x$ e $y$. A dificuldade aqui não é escala desigual, é trajetória curva.

Rodando os seis, mesmo orçamento de 2000 passos, `lr` ajustado por método:

| Otimizador | distância ao mínimo |
|---|---|
| SGD | 0.82 (não saiu do lugar) |
| **Momentum** | **0.0003** |
| **Nesterov** | **0.0002** |
| RMSProp | 0.12 |
| Adam | 0.19 |

Momentum e Nesterov **venceram** os métodos adaptativos, por uma margem
grande. A razão: no fundo do vale, o gradiente aponta consistentemente na
mesma direção geral por muitos passos seguidos — exatamente o regime em
que acumular direção (Momentum) ganha mais do que normalizar escala por
coordenada (Adam/RMSProp), porque aqui o desafio nunca foi escala desigual
entre eixos. "Adam é sempre a escolha segura" é folclore, não teorema — a
geometria do problema decide, e às vezes decide contra o que todo blog de
deep learning sugere de cabeça.

---

## Newton e BFGS: rápido em iterações, caro por iteração

$$
\theta \leftarrow \theta - H^{-1}\nabla\theta
$$

Reaproveita `gradient()` e `hessian()` do Módulo 2 direto — nenhuma
matemática nova.

### Um invariante exato

Numa quadrática pura, a Hessiana é constante: o passo de Newton **é** o
mínimo, não uma aproximação dele. Testado: converge em exatamente 1
iteração, resultado exato até $10^{-8}$.

### No Rosenbrock: 7 iterações contra 2000

Newton convergiu no Rosenbrock em **7 iterações**, até precisão de
máquina ($3.5\times10^{-16}$ de distância). Momentum precisou de 2000
passos pra chegar a $0{,}0003$. Segunda ordem "sabe" a curvatura local e
não precisa tatear — mas cada iteração custa montar e resolver um sistema
$n\times n$.

BFGS (quasi-Newton, aproxima $H^{-1}$ só com gradientes, nunca monta a
Hessiana verdadeira) convergiu em 35 iterações — mais que Newton, menos
que milhares de primeira ordem, e validado contra
`scipy.optimize.minimize(method='BFGS')`, diferença $<10^{-4}$.

### Por que "morre" em alta dimensão — medido, não assumido

Contagem de chamadas a `gradient()` necessárias pra montar a Hessiana via
diferenças finitas, determinístico (não depende de hardware):

| $n$ | chamadas a `gradient()` | tamanho da Hessiana |
|---|---|---|
| 5 | 10 (=2n) | 25 floats |
| 20 | 40 | 400 floats |
| 50 | 100 | 2.500 floats |
| 100 | 200 | 10.000 floats |
| 300 | 600 | 90.000 floats — **3,25s** só pra montar, neste motor |

Um passo de Adam ou SGD, em qualquer dimensão: **1** chamada a
`gradient()`, memória $O(n)$. A Hessiana custa $O(n)$ chamadas de
gradiente pra construir (mesmo com autodiff exato, não só diferenças
finitas — é uma propriedade estrutural, não uma limitação desta
implementação), $O(n^2)$ de memória pra guardar, e resolver o sistema
linear resultante custa $O(n^3)$ com métodos diretos — o mesmo cubo do
número de condição que apareceu no Módulo 1 quando $X^\top X$ elevava
$\kappa$ ao quadrado, agora aparecendo como custo computacional bruto, não
erro numérico.

Uma rede com 100 parâmetros já leva segundos só pra montar a Hessiana
neste motor sem otimização. Um modelo real tem milhões a bilhões de
parâmetros — $O(n)$ gradientes e $O(n^2)$ de memória, sozinhos, já
inviabilizam o método antes mesmo de cogitar o $O(n^3)$ do sistema linear.
É por isso que todo otimizador de deep learning é de primeira ordem: não é
escolha de gosto, é a única classe de método que sobrevive à escala.

---

## Fechando o Módulo 4

Momentum acelera mas pode desestabilizar; Nesterov corrige a
instabilidade sem abrir mão da aceleração; RMSProp equaliza escala entre
parâmetros, à custa de perder o acúmulo direcional; Adam junta os dois,
precisando de correção de viés pra não distorcer os primeiros passos;
AdamW corrige um acoplamento indesejado entre regularização e adaptação
que o próprio Adam introduzia sem avisar. Rosenbrock lembra que nenhum
desses métodos vence sempre — a geometria do problema decide. E Newton
mostra o outro extremo do espectro inteiro: convergência em poucas
iterações, ao custo de um crescimento que simplesmente não cabe em escala
real. Seis trade-offs, cada um medido com números, não assumido por
reputação.
