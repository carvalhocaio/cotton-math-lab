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

## O que vem a seguir

O próximo passo é estudar a decomposição QR, inicialmente construída por Gram-Schmidt e, posteriormente, pela versão numericamente mais estável baseada em reflexões de Householder. Essa decomposição é a base do algoritmo QR, que substitui a deflação sequencial e se tornou o método padrão para o cálculo de autovalores em matrizes densas.
