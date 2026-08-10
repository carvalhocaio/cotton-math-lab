"""Descida de gradiente estocástica - mínima, só o necessário pro capstone.

Momentum, Nesterov, RMSProp, Adam/AdamW ficam para o Módulo 4
(Otimização), onde cada variante ganha sua própria comparação de
trade-offs. Aqui só a forma mais crua do método: um passo, um sinal.
"""

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
