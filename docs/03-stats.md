# 03 — Probabilidade e Estatística

Estimadores aplicados ao micronaire de um lote de fardos (uma
fazenda/safra): MLE como ponto de partida, MAP como o contraponto que
mostra onde um prior ajuda de verdade. Cada estimador é validado por pelo
menos duas rotas independentes — nunca só contra uma biblioteca.

---

## MLE: forma fechada, validada duas vezes

Para dados $\sim \mathcal{N}(\mu, \sigma^2)$, maximizar a log-verossimilhança
tem solução fechada:

$$
\hat\mu = \bar{x}, \qquad \hat\sigma^2 = \frac{1}{n}\sum_i (x_i - \bar{x})^2
$$

Duas validações independentes, nenhuma delas "confiar na fórmula":

**Contra `scipy.stats.norm.fit`** — bateu exato, diferença $0.0$.

**Contra o próprio motor de autodiff do Módulo 2** — maximizando a
log-verossimilhança numericamente via `SGD`, partindo de um chute
deliberadamente ruim ($\mu_0=0$, longe do valor real $\approx4.3$), com
$\sigma$ parametrizado como $\exp(\text{log\_sigma})$ pra garantir
positividade sem precisar de otimização restrita. Convergiu pro mesmo
ponto que a forma fechada, diferença $\sim10^{-9}$. Essa segunda validação
importa mais do que parece: ela prova, ao mesmo tempo, que a forma fechada
está certa **e** que o motor de otimização do Módulo 2 generaliza pra um
problema que não é rede neural nenhuma — é só outra superfície de perda.

### O detalhe que quase todo mundo erra: $\hat\sigma^2$ é enviesado

$\hat\mu$ é sempre não-enviesado, qualquer que seja $n$. $\hat\sigma^2$
**não é** — ele subestima a variância populacional na razão exata
$\frac{n-1}{n}$, medida por Monte Carlo (3000 amostras de tamanho $n=8$):
razão observada $\approx0.857$, esperada $0.875$, dentro do ruído.

Essa é a origem exata da "correção de Bessel" ($/(n-1)$ em vez de $/n$) que
aparece em toda calculadora de variância amostral, `np.var(ddof=1)`
incluído: ela existe especificamente para desfazer o viés que o MLE
introduz. MLE não erra por descuido — ele maximiza a verossimilhança dos
dados observados, e uma amostra sempre varia menos ao redor da sua *própria*
média do que ao redor da média populacional verdadeira (desconhecida). O
`ddof=1` é uma correção post-hoc que o MLE, por definição, não aplica.

---

## MAP: o prior como regularizador que sabe quando parar

Para $\mu$ com $\sigma$ conhecido e prior conjugado
$\mu \sim \mathcal{N}(\mu_0, \tau_0^2)$, a posterior também é Normal, e sua
média (= moda, por simetria) é uma média ponderada por **precisão**
(inverso da variância) entre prior e dados:

$$
\mu_{\text{MAP}} = \frac{\mu_0/\tau_0^2 + n\bar{x}/\sigma^2}{1/\tau_0^2 + n/\sigma^2}
$$

Validado contra busca em grade na posterior real (prior × verossimilhança,
maximizado ponto a ponto numa grade de 20 mil valores de $\mu$) — diferença
$3\times10^{-6}$, limitada só pela resolução da grade.

### Os dois limites, confirmados

Prior quase não-informativo ($\tau_0^2 \to \infty$): $\mu_{\text{MAP}} \to
\bar{x}$, o MLE — bateu com diferença $<10^{-4}$. Prior extremamente
confiante ($\tau_0^2 \to 0$): $\mu_{\text{MAP}} \to \mu_0$, ignorando os
dados por completo — mesma precisão.

### O número que prova o trade-off prometido desde o início do laboratório

Com $n=4$ e um prior só **aproximadamente** certo ($\mu_0=4.2$, verdade
$=4.3$ — não um prior "trapaceando" com a resposta exata), Monte Carlo com
5000 repetições:

| $n$ | MSE(MLE) | MSE(MAP) | redução |
|---|---|---|---|
| 4 | 0.0392 | 0.0197 | **≈50%** |
| 200 | 0.00080 | 0.00079 | ≈2%, dentro do ruído |

A vantagem de MAP não vem de "trapacear" sabendo a resposta — vem de
combinar duas fontes de informação (prior + dados), e com poucos dados a
fonte extra pesa muito. Conforme $n$ cresce, os dados afogam o prior
sozinhos, e a vantagem desaparece — exatamente como deveria: MAP não é
"melhor que MLE" em geral, é melhor **especificamente no regime de dado
escasso onde MLE tem alta variância**.

### A conexão que vale a carreira inteira

Um prior Normal centrado em zero sobre os pesos de uma rede neural **é**
regularização L2 — a derivação bayesiana do MAP com esse prior específico
produz exatamente o termo $\lambda\|\theta\|^2$ que aparece em toda função
de perda regularizada. O que no Módulo 4 vai parecer um "truque de
engenharia" (adicionar um termo de penalidade pra evitar overfitting) é,
visto por este ângulo, a mesma matemática que acabamos de validar aqui:
um prior informativo reduzindo variância às custas de um pouco de viés,
mais útil quanto menos dados houver.

---

## Fio condutor do módulo até aqui

MLE responde "que parâmetro torna os dados observados mais prováveis,
sem nenhuma outra informação". MAP responde a mesma pergunta admitindo
que você já sabia alguma coisa antes de olhar os dados. Nenhum dos dois é
"mais certo" — são respostas a perguntas diferentes, e a escolha certa
depende de quanto dado você tem e quanto confia no que já sabia.
