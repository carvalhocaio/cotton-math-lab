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

## Beta-Binomial: o par natural do Normal-Normal, para dados binários

Mesma lógica do MAP normal — combinar prior e dados —, mas pra uma
proporção em vez de uma média contínua. Prior $p \sim \text{Beta}(\alpha,
\beta)$, observação: $k$ fardos fora de especificação em $n$ inspecionados,
$\sim \text{Binomial}(n, p)$. A posterior é exata, sem nenhuma aproximação:

$$
p \mid k, n \;\sim\; \text{Beta}(\alpha + k,\; \beta + n - k)
$$

Diferente do Normal-Normal do ciclo anterior — que só é exato se $\sigma$
for conhecido —, Beta-Binomial é conjugado **incondicionalmente**: não há
nenhum parâmetro extra que precise ser assumido fixo pra conjugação
funcionar.

### Validação

Contra busca em grade (prior × verossimilhança binomial, integrada
numericamente numa grade de 50 mil pontos): diferença $0.0$ no cenário
testado (6 fora de spec em 40 inspecionados). Contra `scipy.stats.beta`:
exato. E os dois limites de sempre se confirmam: prior quase-flat
($\alpha,\beta \to 0$) faz a média posterior convergir pro MLE $k/n$; um
prior concentrado ($\alpha+\beta$ grande) com poucos dados observados
($n=3$) puxa a média posterior de volta pra perto da média do prior,
resistindo a uma amostra pequena e ruidosa.

### A propriedade que importa mais que a fórmula: atualização sequencial

Atualizar a posterior em dois lotes — primeiro com 4 sucessos em 20
ensaios, depois com 2 em 20 — dá **exatamente** a mesma posterior que
atualizar de uma vez com os 6 em 40 combinados. Não é aproximadamente
igual: os parâmetros batem bit a bit.

Isso não é uma curiosidade de implementação — é a propriedade que torna
inferência bayesiana genuinamente **incremental**. A posterior da semana
passada *é* o prior desta semana. Você pode atualizar a crença sobre a
proporção de fardos fora de especificação a cada novo lote inspecionado,
sem nunca precisar reprocessar o histórico inteiro, e o resultado final
não depende de ter processado tudo de uma vez ou em cinquenta pedaços ao
longo de cinquenta semanas. Um sistema de MLE puro, recalculado do zero a
cada lote, não tem essa propriedade de graça — cada recálculo é
independente do anterior, e "lembrar" exige guardar todo o histórico bruto,
não só dois números ($\alpha$, $\beta$).

### Conectando os dois conjugados do módulo

Normal-Normal e Beta-Binomial resolvem o mesmo problema estrutural —
combinar uma crença prévia com evidência nova, ponderando pela confiança
relativa de cada uma — em dois domínios diferentes: um contínuo (a média de
uma distribuição normal), outro discreto (uma proporção). A forma de
combinar difere (média ponderada por precisão vs. soma de contagens), mas
a lógica é idêntica, e é a mesma lógica que aparece de novo, disfarçada de
"regularização" ou "suavização de Laplace", em praticamente todo canto do
aprendizado de máquina.

---

## Fio condutor do módulo até aqui

MLE responde "que parâmetro torna os dados observados mais prováveis,
sem nenhuma outra informação". MAP responde a mesma pergunta admitindo
que você já sabia alguma coisa antes de olhar os dados. Nenhum dos dois é
"mais certo" — são respostas a perguntas diferentes, e a escolha certa
depende de quanto dado você tem e quanto confia no que já sabia.

---

## Bootstrap: quando não existe conjugado à mão

Os dois ciclos anteriores dependeram de conjugação exata — Normal-Normal,
Beta-Binomial. Mas a maioria das estatísticas de interesse não tem
conjugado nenhum: mediana, correlação, razão de variâncias, o coeficiente
de uma regressão. Bootstrap resolve isso sem assumir nada sobre a forma da
estatística.

A ideia inteira em uma frase: reamostrar os dados observados **com
reposição**, muitas vezes, recalcular a estatística de interesse em cada
reamostra, e usar os percentis dessa distribuição simulada como intervalo.
Não há fórmula de erro-padrão nenhuma — a variabilidade amostral é
estimada empiricamente, tratando a amostra observada como se fosse a
própria população.

### Validação em duas frentes

Contra `scipy.stats.bootstrap`, com a mesma semente: bate essencialmente
exato, tanto para a média quanto para a **mediana de uma distribuição
exponencial** (assimétrica de propósito — sem fórmula fechada simples de
erro-padrão pra mediana, exatamente o caso em que bootstrap importa).

Mais rigoroso: cobertura empírica medida por repetição do experimento
inteiro. Um IC bootstrap de 95%, repetido 1000-2000 vezes com amostras
frescas, capturou a verdade em **93-94%** das vezes — não os 95% exatos.
Isso não é bug: o método percentil tem viés de sub-cobertura conhecido,
mais pronunciado em amostras modestas (existe uma correção — *bias-corrected
and accelerated*, BCa — fora do escopo deste ciclo). O número importa mais
que a intuição: "parece que devia dar 95%" não é o mesmo que medir que dá.

---

## IC vs. intervalo de credibilidade: a confusão mais comum em estatística

Duas construções que respondem perguntas **diferentes**, mesmo quando o
número de saída parece o mesmo tipo de coisa (um intervalo, um nível de
confiança de 95%).

**Intervalo de confiança (Wilson, pra uma proporção)** não usa nenhum
prior — é a inversão de um teste de hipótese: o conjunto de valores $p_0$
para os quais um teste de score não seria rejeitado. A garantia é sobre o
**procedimento**: repetindo o experimento inteiro muitas vezes, 95% dos
intervalos construídos vão conter o $p$ verdadeiro. Para um único intervalo
já calculado, $p$ ou está lá dentro ou não está — não há "95% de chance"
sobre esse intervalo específico, porque no enquadramento frequentista $p$ é
fixo, não aleatório.

**Intervalo de credibilidade** vem dos quantis da posterior Beta. Aqui $p$
**é** tratado como variável aleatória (dado os dados observados), e a
afirmação é direta: "95% de probabilidade posterior de $p$ estar aqui,
dado esses dados e este prior". A condição extra — "e este prior" — é
exatamente o que a leitura popular incorreta do IC costuma esquecer.

### A prova, com números medidos

Um único exemplo concreto já mostra a divergência: com $k{=}6$ fora de
especificação em $n{=}40$ inspecionados, e um prior forte centrado em 0.20
(fixado **antes** de ver os dados):

| | limite inferior | limite superior |
|---|---|---|
| IC (Wilson, 95%) | 0.071 | 0.291 |
| Credível (prior forte) | 0.145 | 0.244 |
| Credível (prior fraco) | 0.071 | 0.292 |

O IC não muda nunca, porque não usa prior nenhum. O credível com prior
forte é visivelmente mais estreito e deslocado — a crença prévia pesou. O
credível com prior fraco praticamente coincide com o IC: **quando o prior
não pesa, os dois enquadramentos convergem numericamente**, mesmo
continuando filosoficamente distintos.

A parte que realmente separa os dois é a **cobertura sob repetição**, com
a verdade fixada em $p{=}0.15$ e um prior **descasado** (centrado em 0.20,
fixado antes de qualquer dado):

| Método | Cobertura medida (nominal: 95%) |
|---|---|
| IC (Wilson) | 96.3% |
| Credível, prior descasado | **74.1%** |
| Credível, prior fraco | 96.3% |

O IC mantém a garantia de 95% **de qualquer jeito** — é uma propriedade da
construção, não depende de nada estar "certo" sobre $p$. O intervalo de
credibilidade com um prior errado **não** mantém essa garantia: caiu 21
pontos percentuais abaixo do nominal. Isso não invalida a leitura
bayesiana — o intervalo continua correto como afirmação de crença
posterior, dado aquele prior específico. O que fica provado é que essa
afirmação de crença **não é a mesma coisa** que a garantia frequentista de
cobertura repetida, e tratar uma como se fosse a outra é exatamente o erro
mais comum que existe em como se comunica estatística.

### A régua pra decidir qual usar

Nenhum dos dois é "mais certo" — respondem perguntas diferentes. Use IC
quando a pergunta é "que garantia esse procedimento tem, sob repetição,
sem eu precisar assumir nada prévio sobre o parâmetro". Use intervalo de
credibilidade quando você **tem** uma crença prévia genuína (histórico de
safras anteriores, conhecimento de domínio sobre a descaroçadora) e quer
uma afirmação de probabilidade direta sobre o parâmetro, condicionada
nessa crença — mas então a validade da afirmação depende inteiramente de
quão razoável foi o prior escolhido, algo que o método em si nunca vai te
avisar se você errou.

---

## Fechando o Módulo 3

MLE responde "o que os dados sozinhos sugerem". MAP e Beta-Binomial
respondem a mesma pergunta admitindo uma crença prévia, e mostram
(com números, não só teoria) que essa crença ajuda mais exatamente quando
os dados são escassos. Bootstrap dispensa fórmula fechada quando nenhum
conjugado existe. E IC vs. credível fecha o módulo com o ponto mais
aplicável de todos: dois números que parecem a mesma coisa podem carregar
garantias matemáticas completamente diferentes — e a diferença só aparece
quando alguém mede, não quando alguém assume.
