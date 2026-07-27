# 00 — Gerador Sintético de Dados HVI

## Por que dados sintéticos, e não dados reais da empresa?

O uso de dados sintéticos oferece uma vantagem pedagógica importante: todos os parâmetros do modelo são conhecidos por construção. Como os valores utilizados para gerar os dados são previamente definidos, é possível verificar se os algoritmos implementados conseguem recuperar esses mesmos parâmetros, avaliando objetivamente sua corretude.

Em dados reais, os parâmetros verdadeiros são desconhecidos. Assim, mesmo que uma estimativa pareça plausível, não existe uma referência exata para medir sua precisão. Já em dados sintéticos, pode-se comparar diretamente o valor estimado com o valor utilizado na geração dos dados, permitindo validar tanto a implementação quanto o comportamento estatístico do método.

Essa estratégia torna-se especialmente importante no Módulo 3, em que a estimação por Máxima Verossimilhança (MLE) deve recuperar os parâmetros utilizados na geração da distribuição. Se as estimativas convergem para os valores conhecidos, há uma evidência concreta de que a implementação está correta antes de sua aplicação em dados reais.

---

## Por que Cholesky, e não `rng.multivariate_normal`?

Fato: se $z \sim \mathcal{N}(0, I)$ e $\Sigma = LL^\top$ (Cholesky), então

$$
\mu + Lz \sim \mathcal{N}(\mu, \Sigma).
$$

Embora a função `rng.multivariate_normal` produza o mesmo resultado, implementar explicitamente a fatoração de Cholesky torna o processo de geração dos dados transparente. Em vez de utilizar uma função de alto nível como uma "caixa-preta", fica evidente que a geração de variáveis correlacionadas consiste em transformar uma distribuição normal, padrão por meio da matriz $L$, que funciona como uma "raiz quadrada" da matriz de covariância.

Além do ganho didático, essa fatoração reaparece em diversos algoritmos estudados posteriormente. No Módulo 4, ela é utilizada na construção de pré-condicionadores para melhorar a estabilidade e a velocidade de métodos numéricos. Já no Módulo 5, a decomposição de Cholesky é empregada em cálculos envolvendo distribuições gaussianas, como a divergência de Kullback-Leibler (KL), em que são necessárias operações com determinantes e inversas de matrizes de covariância.

Dessa forma, implementar manualmente a transformação ajuda a compreender um conceito que será reutilizado ao longo de todo o laboratório.

---

## A transformação de Fisher (variance-stabilizing transform)

Fato: o coeficiente de Pearson vive em $[-1,1]$ e sua variância amostral **depende do próprio valor**. A transformação de Fisher estabiliza essa variância:

$$
z = \operatorname{arctanh}(r)
= \frac{1}{2}\ln\left(\frac{1+r}{1-r}\right),
\qquad
\operatorname{SE}(z)\approx\frac{1}{\sqrt{n-3}}.
$$

Utilizar uma tolerância absoluta, como $|r_{\text{emp}} - r_{\text{true}}| < 0{,}02$, não é uma boa estratégia porque a variabilidade da estimativa depende do próprio valor da correlação. Correlações próximas de $\pm1$ apresentam naturalmente menor variância do que correlações quase inexistentes. Assim, a mesma diferença absoluta pode representar um erro esperado em um caso e um erro estatisticamente improvável em outro.

A transformação de Fisher resolve esse problema ao converter a correlação para uma escala em que a variância é aproximadamente constante, com erro-padrão dependente apenas do tamanho da amostra. Isso permite comparar estimativas utilizando um mesmo critério estatístico, independentemente do valor verdadeiro da correlação.

Esse princípio aparece em diversos outros contextos da Estatística. O logit transforma probabilidades limitadas ao intervalo $(0,1)$ em uma escala ilimitada, enquanto o logaritmo transforma variáveis estritamente positivas para reduzir assimetria e estabilizar a variância. Da mesma forma, as funções de ligação (*link functions*) dos Modelos Lineares Generalizados (GLMs) utilizam transformações para tornar a modelagem mais adequada às propriedades estatísticas dos dados.

---

## Decisões de design registradas

- **Retorno como `ndarray (n, k)` em vez de `DataFrame`:** mantém a função independente de bibliotecas de análise, reduz overhead e facilita a integração com algoritmos numéricos que operam diretamente sobre arrays do NumPy.

- **`seed` explícita em vez do estado global:** garante reprodutibilidade dos experimentos e evita que diferentes partes do código interfiram entre si ao compartilhar o mesmo gerador de números aleatórios.

- **`eq=False` no `frozen dataclass`:** impede comparações automáticas baseadas em igualdade de arrays, evitando ambiguidades e erros como *"The truth value of an array is ambiguous"*.

- **Ordem das validações (dimensão → positividade → simetria → diagonal → definida positiva):** verifica primeiro condições simples e baratas antes de executar testes computacionalmente mais caros, produzindo mensagens de erro mais claras e específicas.

- **Mensagem de erro informa o menor autovalor:** além de indicar que a matriz não é definida positiva, mostra o quanto ela viola essa condição, facilitando tanto a depuração quanto o ajuste dos parâmetros utilizados.
