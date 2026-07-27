# 01 — Álgebra Linear

## Power Iteration

Itera

$$
v_{k+1} = \frac{Av_k}{\lVert Av_k \rVert}.
$$

O autovalor é estimado pelo quociente de Rayleigh

$$
\lambda = \frac{v^\top A v}{v^\top v},
$$

que, para vetores normalizados, reduz-se a $v^\top A v$.

### Por que converge, e a que velocidade?

O método converge porque o vetor inicial pode ser escrito como uma combinação linear dos autovetores da matriz. Se $A$ possui autovetores $v_1, v_2, \ldots, v_n$, então

$$
v_0 = c_1v_1 + c_2v_2 + \cdots + c_nv_n.
$$

Ao multiplicar sucessivamente por $A$, cada componente é escalada pelo seu respectivo autovalor:

$$
A^k v_0 = c_1\lambda_1^k v_1 + c_2\lambda_2^k v_2 + \cdots + c_n\lambda_n^k v_n.
$$

Se $|\lambda_1| > |\lambda_2| \geq \cdots$, a componente associada ao maior autovalor cresce relativamente mais rápido que as demais. Após cada normalização, as componentes menores tornam-se cada vez menos significativas, fazendo com que o vetor iterado se aproxime do autovetor dominante.

A velocidade dessa convergência é geométrica e depende da razão

$$
\left|\frac{\lambda_2}{\lambda_1}\right|.
$$

Quanto menor essa razão, mais rapidamente o erro diminui a cada iteração. Isso explica o conceito de **gap espectral**: quando existe uma grande diferença entre o maior e o segundo maior autovalor em módulo, a convergência é muito mais rápida. Esse comportamento é exatamente o observado no teste `test_converges_faster_with_larger_spectral_gap`, em que matrizes com maior separação entre os autovalores convergem em menos iterações.

### Por que o quociente de Rayleigh é quadraticamente preciso?

Observação numérica (do protótipo): com $A$ simétrica $6 \times 6$, o erro do **autovalor** ficou em aproximadamente $10^{-11}$ enquanto o resíduo $\lVert Av - \lambda v \rVert$ (governado pelo erro do **autovetor**) ficou em aproximadamente $10^{-5}$.

Fato: o autovalor converge em $O(\varepsilon^2)$, enquanto o autovetor converge em $O(\varepsilon)$.

O quociente de Rayleigh possui uma propriedade importante: o erro na estimativa do autovalor é de segunda ordem em relação ao erro do autovetor.

Se o vetor aproximado é escrito como

$$
v = v^* + \varepsilon u,
$$

onde $v^*$ é o autovetor verdadeiro e $\varepsilon$ representa um pequeno erro, a correção de primeira ordem do quociente de Rayleigh desaparece. Isso ocorre porque o autovetor verdadeiro é um ponto estacionário do quociente de Rayleigh, fazendo com que apenas termos proporcionais a $\varepsilon^2$ permaneçam.

**A prova**, para fechar o argumento sem deixar como afirmação solta:

Seja $v = v^* + \varepsilon u$, com $u \perp v^*$ e $\lVert v^* \rVert = 1$. Então

$$
R(v) = \frac{v^\top A v}{v^\top v}
= \frac{\lambda^* + 2\varepsilon\, u^\top A v^* + \varepsilon^2 u^\top A u}
       {1 + \varepsilon^2 \lVert u \rVert^2}.
$$

Como $v^*$ é autovetor, $Av^* = \lambda^* v^*$, e como $u \perp v^*$, o termo
cruzado se anula exatamente:

$$
2\varepsilon\, u^\top A v^* = 2\varepsilon \lambda^* (u^\top v^*) = 0.
$$

Não é uma aproximação — é uma identidade que vale para qualquer $\varepsilon$.
Sobra apenas

$$
R(v) = \lambda^* + \varepsilon^2\left(u^\top A u - \lambda^* \lVert u \rVert^2\right) + O(\varepsilon^4).
$$

O termo de primeira ordem em $\varepsilon$ nunca existiu: ele desaparece porque
$v^*$ é ponto estacionário do quociente de Rayleigh, não por coincidência
numérica. É essa identidade que garante, de primeiros princípios, que um erro
$O(\varepsilon)$ no autovetor produz um erro $O(\varepsilon^2)$ no autovalor.

Como consequência, enquanto o erro do autovetor é da ordem de

$$
O(\varepsilon),
$$

o erro do autovalor é da ordem de

$$
O(\varepsilon^2).
$$

Esse comportamento explica a observação experimental: mesmo com um resíduo em torno de $10^{-5}$, o erro do autovalor já atingia aproximadamente $10^{-11}$. Na prática, isso significa que uma aproximação razoável do autovetor já produz uma estimativa extremamente precisa do autovalor. Essa é uma das principais razões pelas quais o quociente de Rayleigh é amplamente utilizado em algoritmos para cálculo de autovalores.

---

## Deflação de Hotelling

Após extrair $(\lambda_1, v_1)$, constrói-se

$$
A' = A - \lambda_1 v_1v_1^\top.
$$

Em seguida, aplica-se novamente o método da potência.

### Por que funciona?

A matriz $v_1v_1^\top$ representa a projeção sobre a direção do primeiro autovetor. Ao multiplicá-la por $\lambda_1$ e subtrair esse termo de $A$, elimina-se exatamente a contribuição correspondente ao autovalor dominante.

De fato,

$$
\begin{aligned}
A'v_1
&= (A-\lambda_1v_1v_1^\top)v_1 \\
&= \lambda_1v_1-\lambda_1v_1 \\
&= 0.
\end{aligned}
$$

Assim, o primeiro autovalor passa a ser zero.

Para qualquer outro autovetor $v_i$, como a matriz é simétrica, seus autovetores são ortogonais entre si. Portanto,

$$
v_1^\top v_i = 0.
$$

Logo,

$$
\begin{aligned}
A'v_i
&= Av_i-\lambda_1v_1(v_1^\top v_i) \\
&= \lambda_i v_i.
\end{aligned}
$$

Ou seja, os demais autovalores e autovetores permanecem inalterados. Dessa forma, o próximo maior autovalor torna-se dominante, permitindo sua extração pela próxima execução do método da potência.

### O trade-off honesto: por que você NÃO usaria isso na prática

Observação numérica: no protótipo, os autovalores bateram o NumPy até $10^{-14}$, mas a ortogonalidade dos autovetores recuperados caiu para aproximadamente $10^{-7}$ — os últimos vetores são visivelmente menos precisos que os primeiros.

Embora a deflação de Hotelling funcione muito bem para matrizes pequenas, ela apresenta um problema importante: os erros numéricos acumulam-se a cada etapa.

Na prática, o primeiro autovetor nunca é calculado exatamente. Assim, a matriz de deflação é construída utilizando uma aproximação de $v_1$. Como consequência, a contribuição do primeiro autovalor não é removida de forma perfeita, introduzindo pequenos erros na nova matriz. Esses erros passam a influenciar o cálculo do segundo autovetor, que por sua vez gera uma nova deflação também imperfeita. Esse processo continua sucessivamente, acumulando erros ao longo das iterações.

Esse fenômeno explica por que, no protótipo, os autovalores continuaram extremamente precisos (erro próximo de $10^{-14}$), enquanto a ortogonalidade dos autovetores recuperados caiu para aproximadamente $10^{-7}$. Os últimos autovetores acabam sendo significativamente menos precisos que os primeiros.

Em matrizes pequenas, como uma matriz $6 \times 6$, esse efeito é praticamente irrelevante. Entretanto, em problemas reais envolvendo milhares de dimensões, o acúmulo de erros pode comprometer seriamente a qualidade dos autovetores obtidos.

Por esse motivo, a deflação sequencial é raramente utilizada em aplicações de grande porte. Na prática, emprega-se o algoritmo QR, que realiza transformações ortogonais de similaridade preservando simultaneamente toda a estrutura espectral da matriz. Como essas transformações mantêm a ortogonalidade de forma muito mais estável numericamente, o algoritmo QR evita o acúmulo progressivo de erros observado na deflação de Hotelling e produz todos os autovalores e autovetores com alta precisão.

---

## Decomposição QR: Gram-Schmidt vs. Householder

Toda matriz $A \in \mathbb{R}^{m\times n}$ ($m \geq n$, colunas linearmente
independentes) se decompõe como $A = QR$, com $Q$ de colunas ortonormais e
$R$ triangular superior. Os dois métodos abaixo chegam no mesmo resultado
teórico por caminhos numericamente muito diferentes.

### Gram-Schmidt clássico

Constrói $Q$ coluna a coluna: a $j$-ésima coluna de $A$ tem removida sua
projeção sobre todas as colunas anteriores de $Q$, e o resultado é
normalizado.

$$
v_j = a_j - \sum_{i<j} (q_i^\top a_j)\, q_i, \qquad q_j = \frac{v_j}{\lVert v_j \rVert}
$$

O problema não é a fórmula — é a aritmética de ponto flutuante. Quando duas
colunas de $A$ são quase paralelas, $v_j$ é uma **diferença entre duas
quantidades quase iguais** ($a_j$ e sua projeção). Esse tipo de subtração é o
caso clássico de *cancelamento catastrófico*: os dígitos significativos que
sobram depois da subtração vêm majoritariamente do erro de arredondamento de
cada termo, não do sinal real. O erro de uma projeção contamina a próxima
coluna, que contamina a seguinte — e a ortogonalidade de $Q$ degrada de forma
acumulativa e silenciosa, sem que a fatoração pareça "quebrada": $QR$ ainda
reconstrói $A$ com precisão de máquina, só $Q$ deixa de ser ortogonal de
verdade.

O fixture do teste torna isso concreto: com três colunas quase paralelas
(diferença de $10^{-7}$ entre elas), o erro de ortogonalidade de $Q$ salta de
precisão de máquina para $\approx 1.9\times10^{-2}$ — quatro ordens de
grandeza de degradação, e a matriz de teste nem é patologicamente extrema.

### Reflexões de Householder

Em vez de projetar e subtrair, cada passo aplica uma **reflexão ortogonal**
$H = I - 2vv^\top$ (com $\lVert v \rVert = 1$) escolhida para zerar tudo
abaixo da diagonal na coluna atual.

A diferença estrutural é o que importa: uma reflexão de Householder é uma
**isometria exata por construção** — preserva norma e ângulo entre quaisquer
vetores, não como resultado de uma conta bem-sucedida, mas porque
$H^\top H = I$ é uma identidade algébrica, verdadeira a cada passo
independentemente de quão mal-condicionada a matriz de entrada seja. Não há
subtração de quantidades quase iguais escondida no processo — o cancelamento
catastrófico simplesmente não tem onde acontecer.

O mesmo fixture prova isso: erro de ortogonalidade $\approx 1.4\times10^{-15}$,
precisão de máquina, na mesma matriz onde o Gram-Schmidt clássico falhou.

### O padrão que já apareceu antes

Esta é a mesma estrutura de trade-off da deflação de Hotelling, e vale
nomear o padrão geral: **métodos que operam por diferenças sucessivas (GS,
deflação) acumulam erro de arredondamento a cada passo; métodos que operam
por transformações exatamente ortogonais a cada passo (Householder, e o
algoritmo QR que segue) não acumulam, porque cada passo é uma
isometria por definição, não por sorte numérica.**

Isso não é coincidência de dois exemplos — é o critério de estabilidade
numérica que separa "método didático" de "método de produção" em quase toda
álgebra linear numérica: prefira transformações ortogonais as projeções
sempre que a estabilidade importar mais que a simplicidade da fórmula.

### Consequência prática

Isso é também por que o algoritmo QR para autovalores (próximo ciclo)
substitui a deflação de Hotelling: ele usa exatamente estas reflexões de
Householder para reduzir a matriz e depois itera $A_{k+1} = R_k Q_k$
(fatoração e produto na ordem trocada) sem nunca acumular o erro que a
deflação sequencial carrega.

---

## Algoritmo QR: fechando o módulo

A cada passo, fatora $A_k = Q_k R_k$ e recompõe na ordem trocada:
$A_{k+1} = R_k Q_k$.

### A identidade que faz tudo funcionar

$$
A_{k+1} = R_k Q_k = Q_k^\top (Q_k R_k) Q_k = Q_k^\top A_k Q_k.
$$

Cada passo é uma **transformação de similaridade ortogonal**. Isso importa
porque similaridade preserva o espectro exatamente: $A_k$ e $A_{k+1}$ têm os
mesmos autovalores, sempre, para qualquer $k$. O algoritmo não "calcula" os
autovalores — ele só muda a base em que a matriz é representada, até a base
escolhida ser aquela em que a matriz já é diagonal. Nessa base, os
autovalores estão, por definição, na diagonal.

### O trade-off que amarra os três ciclos do módulo

O algoritmo QR resolve exatamente o problema que a deflação de Hotelling
tinha: como usa Householder a cada fatoração, não há subtração de
quantidades quase iguais, não há cancelamento catastrófico, e a
ortogonalidade dos autovetores recuperados não degrada — o teste
`test_no_orthogonality_degradation_across_deflation_like_use` prova isso até
$10^{-8}$ mesmo após reduzir 8 dimensões.

Mas ele herda, sem disfarce, o problema de velocidade da power iteration.
Estruturalmente, o QR algorithm é uma **iteração de subespaço simultânea** —
em vez de perseguir um único autovetor dominante, persegue um subespaço
inteiro ao mesmo tempo, mas o mecanismo de convergência é o mesmo: geométrico
na razão $|\lambda_{k+1}/\lambda_k|$ entre autovalores consecutivos. O
experimento no protótipo tornou isso concreto: gap largo ($\lambda_1{=}10$,
$\lambda_2{=}1$) convergiu em 32 iterações; o mesmo tamanho de matriz com gap
estreito ($\lambda_1{=}10$, $\lambda_2{=}9.999$) não convergiu nem em 3000.

Não existe almoço grátis aqui: você trocou "impreciso perto do fim do
espectro" por "lento perto de autovalores próximos". Nenhum dos três métodos
deste módulo — power iteration, deflação, QR sem shift — escapa de um dos
dois problemas.

### O que fica de fora, por honestidade

A correção de produção para a lentidão é o **shift de Wilkinson**: subtrair
de $A_k$ uma estimativa do autovalor mais próximo antes de cada fatoração,
o que acelera a convergência de linear para cúbica — poucas iterações bastam
mesmo com gaps estreitos. Fora do escopo deste módulo implementar, mas vale
registrar que o problema tem solução conhecida, e qual é o princípio dela.

### Fechando o ciclo do Módulo 1

Três métodos, um só objetivo (autovalores), três trade-offs diferentes:
power iteration é simples, mas só dá o autovalor dominante; deflação estende
para o espectro inteiro, mas acumula erro; QR corrige o erro, mas herda a
lentidão. É exatamente esse tipo de mapa — não "qual método é o melhor", mas
"qual dor cada método troca por qual outra" — que separa julgar de
primeiros princípios de decorar qual função chamar.
