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
