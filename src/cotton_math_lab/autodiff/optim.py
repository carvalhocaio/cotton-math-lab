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
