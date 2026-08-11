# 05 — Teoria da Informação

Entropia, cross-entropy e KL primeiro em versão discreta (distribuições
sobre categorias — faixas de qualidade de fardos), depois contínua (KL
entre densidades). O ciclo central do módulo é a assimetria de KL — o
motivo pelo qual "distância" é a palavra errada pra descrever o que ela
mede.

---

## Entropia, cross-entropy, KL: uma identidade, não três conceitos soltos

$$
H(p) = -\sum_x p(x)\log p(x), \qquad H(p,q) = -\sum_x p(x)\log q(x)
$$

$$
D_{KL}(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)} = H(p,q) - H(p)
$$

A terceira fórmula não é uma coincidência — é uma identidade, testada
diretamente: `H(p,q) = H(p) + D_KL(p‖q)`, bate exato. Cross-entropy é o
custo médio de codificar amostras de $p$ usando um código otimizado para
$q$; entropia é o custo mínimo possível (código otimizado para o próprio
$p$); KL é a diferença — o quanto se paga a mais por usar o código errado.

### Duas propriedades que sustentam tudo o resto do módulo

**Desigualdade de Gibbs**: $D_{KL}(p\|q) \geq 0$ sempre, para qualquer par
de distribuições — testada em 5000 pares aleatórios via simplex
(Dirichlet), zero violações. Igualdade só quando $p=q$ exatamente.

**A uniforme maximiza entropia**: entre 3000 distribuições aleatórias
sobre 5 categorias, nenhuma superou $H(\text{uniforme}) = \log 5$. Faz
sentido: entropia mede "quanta incerteza sobra", e nenhuma distribuição
tem mais incerteza sobre $n$ resultados que dar peso igual a todos.

---

## A assimetria de KL: por que "distância" é a palavra errada

$D_{KL}(p\|q) \neq D_{KL}(q\|p)$ em geral — KL não é simétrica, então não é
uma métrica de distância no sentido formal (não satisfaz a desigualdade
triangular nem a simetria). Essa assimetria não é só um detalhe técnico:
ela produz comportamentos **qualitativamente diferentes** quando usada como
objetivo de otimização.

### O experimento: ajustar uma Gaussiana a um alvo bimodal

Alvo: $p(x) = 0.5\,\mathcal{N}(x;-3,1) + 0.5\,\mathcal{N}(x;3,1)$ — dois
modos bem separados. Ajustar $q$ Gaussiana minimizando cada direção de KL.

**Forward KL** ($D_{KL}(p\|q)$, minimizada sobre $q$): tem forma fechada
para $q$ na família exponencial — casamento de momentos, $\mu = E_p[x]$,
$\sigma^2 = \text{Var}_p[x]$. Resultado: $\mu=0$, $\sigma=3.16$ — uma
Gaussiana **larga**, cobrindo os dois modos (e desperdiçando massa no
vale entre eles, onde $p$ é baixo). Essa propriedade de projeção de
momento vale para qualquer família exponencial, não só Gaussiana — é
matemática de forma fechada, não uma coincidência do experimento.

**Reverse KL** ($D_{KL}(q\|p)$, minimizada sobre $q$): sem forma fechada
geral, exige otimização numérica — e o resultado **depende de onde a
otimização começa**:

| Início | $\mu^*$ | $\sigma^*$ | KL final |
|---|---|---|---|
| perto do modo esquerdo | -2.984 | 1.023 | 0.689 |
| perto do modo direito | 2.984 | 1.023 | 0.689 |
| no meio (entre os modos) | 0.000 | 2.777 | 0.834 (pior!) |

Começando perto de qualquer modo, $q$ **colapsa** nele — média e desvio
praticamente replicam um componente da mistura, ignorando o outro por
completo. Começando no meio, a otimização fica presa num mínimo local
pior. Isso não é falha de otimização: é a geometria da superfície de
reverse KL, que tem um mínimo local perto de cada modo de $p$, porque $q$
é penalizada por colocar massa onde $p$ é baixo — mas nunca é obrigada a
cobrir toda a massa de $p$.

### Mean-seeking vs. mode-seeking

A forma resumida que vale carregar: **forward KL "espalha"** (cobre todo o
suporte de $p$, aceita desperdiçar massa em regiões de baixa densidade,
porque o termo $p(x)\log(p(x)/q(x))$ penaliza pesado quando $p(x)>0$ mas
$q(x)\to0$). **Reverse KL "colapsa"** (concentra em um único modo, porque
o termo $q(x)\log(q(x)/p(x))$ só é avaliado onde $q$ põe massa — colocar
massa num vale de baixa densidade de $p$ já é caro o suficiente pra
$q$ preferir evitar completamente).

### Por que isso importa pro resto da carreira

Essa distinção não é curiosidade teórica — é o motivo estrutural por trás
de decisões de design em ML moderno:

- **Destilação de conhecimento** tipicamente minimiza forward KL (a saída
  do professor é $p$, a do aluno é $q$) — quer que o aluno cubra todo o
  comportamento do professor, mesmo em casos raros.
- **DPO e RLHF** frequentemente envolvem reverse KL entre a política
  otimizada e uma política de referência — o objetivo é ficar perto de um
  comportamento específico, não cobrir toda a distribuição de
  possibilidades, e o colapso de modo é, nesse contexto, uma feature
  (foco), não um bug.
- **VAEs** usam reverse KL na regularização do espaço latente
  precisamente porque colapso de modo, ali, produz representações mais
  "decodificáveis" — cobrir todo o espaço de possibilidades produziria um
  posterior difuso demais pra ser útil.

Entender qual direção de KL um método usa — e por quê — é o que separa
"sei que existe uma função de perda chamada KL" de "sei prever que
comportamento essa escolha específica vai produzir".

---

## Informação mútua: o que correlação não vê

$$
I(X;Y) = \sum_{x,y} p(x,y)\log\frac{p(x,y)}{p(x)p(y)}
$$

Estimada via binning: discretiza duas variáveis contínuas em faixas,
calcula o histograma conjunto, aplica a fórmula acima sobre as
probabilidades empíricas. Validado contra `sklearn.metrics.mutual_info_score`
usando os mesmos limites de faixa nos dois cálculos — diferença
$2\times10^{-16}$, precisão de máquina.

### O experimento que prova o valor da ferramenta

$Y = X^2$, com $X$ uniforme em $[-1,1]$ (simétrico em torno de zero):

| Medida | Valor |
|---|---|
| Correlação de Pearson | 0.006 (essencialmente zero) |
| Informação mútua | 1.96 (dependência forte) |

A dependência entre $X$ e $Y$ é **perfeita e determinística** — sabendo
$X$, você sabe $Y$ exatamente. Mas é simétrica em torno de zero, então a
covariância (o numerador de Pearson) cancela: valores positivos e
negativos de $X$ contribuem com sinais opostos, e a soma dá
aproximadamente zero. Pearson literalmente não consegue ver essa
dependência, porque só é sensível à componente **linear** de qualquer
relação. Informação mútua não tem esse ponto cego — mede dependência
estatística de qualquer forma, linear ou não.

### Aplicação nos dados HVI reais

Comparando um par de features correlacionadas por construção no gerador
do Módulo 0 (`uhml`, `uniformity`, correlação 0.55) contra um par
independente por construção (`micronaire`, `rd`, correlação 0):

$$
\text{MI(uhml, uniformity)} = 0.208 \qquad \text{MI(micronaire, rd)} = 0.026
$$

Quase 8× maior — informação mútua recuperou a estrutura de dependência
real que o gerador plantou, sem nenhuma suposição de linearidade. Para
seleção de features em dados HVI de verdade, onde relações entre
parâmetros físicos raramente são puramente lineares (maturidade da fibra
afeta micronaire e resistência de formas não necessariamente aditivas),
essa é uma vantagem concreta sobre olhar só a matriz de correlação.

---

## JS como detector de drift: a aplicação que antecipa monitoramento de produção

KL tem dois problemas pra servir de "quão diferente é a safra desta
semana da safra de referência": não é simétrica (KL(safra_nova‖base) ≠
KL(base‖safra_nova) — qual das duas é "a resposta certa"?), e pode
explodir pra infinito se a safra nova tiver valores fora do que a
distribuição base jamais observou. Jensen-Shannon resolve os dois:

$$
D_{JS}(p,q) = \tfrac{1}{2}D_{KL}(p\|m) + \tfrac{1}{2}D_{KL}(q\|m), \qquad m = \tfrac{p+q}{2}
$$

Simétrica por construção (testado exato), limitada entre 0 e $\log 2$
(testado nos dois extremos — distribuições idênticas dão 0, distribuições
com suporte completamente disjunto dão $\log 2$ exato). Validado contra
`scipy.spatial.distance.jensenshannon` — que devolve a *raiz* da
divergência (a distância JS), então a comparação eleva ao quadrado antes.

### A prova de que funciona como detector, não só como métrica

Três safras simuladas via o gerador do Módulo 0: uma base, uma segunda
amostra da **mesma** distribuição (só ruído de amostragem — sementes
diferentes, sem mudança real), e uma terceira com deslocamento real de
0.6 na média do micronaire (simulando uma safra genuinamente diferente).

| Comparação | $D_{JS}$ |
|---|---|
| Base vs. mesma distribuição (ruído amostral) | 0.0033 |
| Base vs. distribuição deslocada (drift real) | **0.219** |

Quase 65× maior no caso de drift real. `detect_drift()` usa exatamente
essa separação: um threshold (calibrável a partir de dados históricos de
quanto ruído amostral genuíno costuma produzir) decide se a divergência
observada é grande demais pra ser só variação natural entre safras
semelhantes.

Isso é, literalmente, a técnica central de monitoramento de modelos em
produção — detectar quando a distribuição dos dados de entrada mudou o
suficiente pra colocar em dúvida se um modelo treinado no passado ainda é
confiável. A Fase 3 do roteiro (avaliação e MLSecOps) vai reencontrar essa
exata ferramenta, só que aplicada a *features de um modelo em produção*
em vez de *parâmetros HVI de uma safra* — o método não muda, só o domínio.

---

# Fechando a Fase 1.1: Fundamentos Matemáticos

Cinco módulos, um só objetivo: fluência que sobrevive fora do contexto em
que foi aprendida.

**Álgebra Linear** — três métodos de autovalor, cada um trocando uma dor
por outra (velocidade, precisão, ou ambas); PCA revelando que a "forma
óbvia" de decompor dados (via covariância) perde precisão exatamente
quando mais precisa funcionar, e SVD direto na matriz de dados resolvendo
isso ao custo de nunca elevar o número de condição ao quadrado.

**Cálculo / Autodiff** — um motor de ~300 linhas que deriva qualquer
composição de seis primitivas, os dois modos (reverso, forward) trocando
custo de $m$ por custo de $n$ dependendo da forma do problema, e um
classificador real treinado do zero que bateu acurácia com o `sklearn`.

**Probabilidade e Estatística** — MLE errando de forma previsível
(variância enviesada por $(n-1)/n$), MAP consertando exatamente quando os
dados são escassos, e a prova mais aplicável da fase inteira: intervalo
de confiança e intervalo de credibilidade não são a mesma garantia, e a
diferença custa 21 pontos percentuais de cobertura quando confundida.

**Otimização** — seis otimizadores validados trajetória por trajetória
contra `torch`, Rosenbrock desmentindo o clichê "Adam é sempre melhor", e
Newton mostrando o preço da convergência rápida: $O(n)$ gradientes e
$O(n^2)$ de memória só pra montar a Hessiana, antes mesmo do $O(n^3)$ de
resolver o sistema — a razão estrutural pela qual todo otimizador de
produção em deep learning é de primeira ordem.

**Teoria da Informação** — entropia, KL e a assimetria que separa
"cobrir toda a distribuição" de "colapsar num modo", com consequências
diretas em como destilação e DPO se comportam; informação mútua vendo
dependência que Pearson não vê; e JS fechando com uma aplicação real de
monitoramento, antecipando a Fase 3.

O padrão que atravessa os cinco módulos, não por coincidência: **a forma
mais direta de resolver um problema numérico raramente é a mais estável,
e a diferença só aparece quando alguém mede** — número de condição ao
quadrado, cancelamento catastrófico, viés de estimador, instabilidade de
otimizador, colapso de modo. Nenhum desses trade-offs foi aceito por
autoridade de livro-texto neste laboratório; todos foram reproduzidos,
medidos, e só então documentados. É essa disposição — desconfiar da
resposta óbvia até ela sobreviver a um teste — que a Fase 1 pretendia
treinar, e é exatamente o que separa ler um paper de julgá-lo.
