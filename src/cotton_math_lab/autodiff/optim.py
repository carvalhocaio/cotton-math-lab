"""Descida de gradiente estocástica - mínima, só o necessário pro capstone.

Momentum, Nesterov, RMSProp, Adam/AdamW ficam para o Módulo 4
(Otimização), onde cada variante ganha sua própria comparação de
trade-offs. Aqui só a forma mais crua do método: um passo, um sinal.
"""

import numpy as np

from cotton_math_lab.autodiff.tensor import Tensor


class SGD:
    """Atualiza cada parâmetro na direção oposta ao seu gradiente.

    θ ← θ - lr·∇θ. É o passo mais simples possível de descida de
    gradiente: nenhuma memória de passos anteriores, nenhuma adaptação de
    taxa por parâmetro — só o sinal local do gradiente, escalado por `lr`.
    """

    def __init__(self, parameters: list[Tensor], lr: float = 0.01):
        self.parameters = parameters
        self.lr = lr

    def step(self) -> None:
        for parameter in self.parameters:
            parameter.data = parameter.data - self.lr * parameter.grad

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()


class Momentum:
    """SGD com momentum clássico (heavy ball).

    v ← momentum·v + ∇θ
    θ ← θ - lr·v

    O gradiente atual se acumula num "vetor de velocidade" que carrega
    memória exponencialmente decrescente de gradientes passados. Na
    direção em que o gradiente aponta consistentemente pro mesmo lado, v
    cresce e acelera o progresso; na direção em que o gradiente oscila de
    sinal a cada passo, as contribuições tendem a se cancelar parcialmente
    ao longo do tempo. Mas repare: v não tem limite superior automático —
    o passo efetivo pode chegar a ~lr/(1-momentum), bem maior que lr puro.
    É esse mesmo mecanismo que acelera convergência numa direção suave e
    que pode desestabilizar numa direção de alta curvatura, dependendo de
    quão perto do limite de estabilidade o `lr` está.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        for parameter, velocity in zip(self.parameters, self.velocity, strict=True):
            velocity[...] = self.momentum * velocity + parameter.grad
            parameter.data = parameter.data - self.lr * velocity

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()


class NesterovMomentum:
    """Momentum de Nesterov — corrige o gradiente ANTES de aplicar o passo,
    usando a velocidade que já se acumulou.

    v ← momentum·v + ∇θ
    θ ← θ - lr·(∇θ + momentum·v)

    A formulação clássica de Nesterov calcula o gradiente numa posição
    "futura" (θ - lr·momentum·v), avaliando a função ali antes de dar o
    passo — uma segunda passada forward por iteração. O PyTorch (e esta
    implementação) usa uma reformulação algébrica equivalente que evita
    essa segunda avaliação: soma momentum·v ao gradiente ATUAL antes de
    escalar por lr, chegando no mesmo destino sem o custo extra. Na
    prática, essa correção antecipada reduz o overshoot que faz o
    Momentum clássico oscilar em superfícies mal-condicionadas — desloca
    a fronteira de estabilidade, não é só uma variação cosmética.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        momentum: float = 0.9,
    ):
        self.parameters = parameters
        self.lr = lr
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        for parameter, velocity in zip(self.parameters, self.velocity, strict=True):
            velocity[...] = self.momentum * velocity + parameter.grad
            update = parameter.grad + self.momentum * velocity
            parameter.data = parameter.data - self.lr * update

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()


class RMSProp:
    """Adapta o tamanho do passo por parâmetro via média móvel do
    gradiente ao quadrado — muda de eixo em relação a Momentum: em vez de
    suavizar a DIREÇÃO do passo, reescala sua MAGNITUDE, parâmetro a
    parâmetro.

    v ← α·v + (1-α)·(∇θ)²
    θ ← θ - lr·∇θ / (√v + ε)

    Parâmetros com gradiente historicamente grande (alta curvatura, como
    y na superfície de teste) acumulam v grande e recebem passo efetivo
    MENOR; parâmetros com gradiente historicamente pequeno recebem passo
    efetivo MAIOR. O resultado prático: direções de curvatura muito
    diferente passam a andar em ritmos parecidos, sem precisar caçar um
    `lr` que funcione simultaneamente para as duas — o problema que
    Momentum, no ciclo anterior, não resolvia sozinho.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
    ):
        self.parameters = parameters
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.square_average = [np.zeros_like(p.data) for p in parameters]

    def step(self) -> None:
        for parameter, square_avg in zip(
            self.parameters, self.square_average, strict=True
        ):
            square_avg[...] = self.alpha * square_avg + (1 - self.alpha) * (
                parameter.grad**2
            )
            parameter.data = parameter.data - self.lr * parameter.grad / (
                np.sqrt(square_avg) + self.eps
            )

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()


class Adam:
    """Combina Momentum (média móvel do gradiente) com RMSProp (média
    móvel do gradiente ao quadrado), mais correção de viés — necessária
    porque m₀=v₀=0 enviesa as duas médias em direção a zero nos primeiros
    passos, e os dois enviesam em proporções DIFERENTES (governadas por
    β₁ e β₂ respectivamente), então o desequilíbrio entre eles distorce o
    tamanho do passo se não for corrigido.

    m ← β₁·m + (1-β₁)·∇θ
    v ← β₂·v + (1-β₂)·(∇θ)²
    m̂ ← m / (1-β₁ᵗ),  v̂ ← v / (1-β₂ᵗ)
    θ ← θ - lr·m̂ / (√v̂ + ε)

    Em t=1, a correção cancela exatamente o fator de viés introduzido:
    m̂ = ∇θ e v̂ = (∇θ)², então m̂/√v̂ = sign(∇θ) — o primeiro passo é
    sempre ±lr, não importa a magnitude do gradiente inicial.
    """

    def __init__(
        self,
        parameters: list[Tensor],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.parameters = parameters
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        for parameter, m, v in zip(self.parameters, self.m, self.v, strict=True):
            m[...] = self.beta1 * m + (1 - self.beta1) * parameter.grad
            v[...] = self.beta2 * v + (1 - self.beta2) * (parameter.grad**2)

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            parameter.data = parameter.data - self.lr * m_hat / (
                np.sqrt(v_hat) + self.eps
            )

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.zero_grad()
