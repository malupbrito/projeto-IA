#Projeto YouBot – Aprendizagem de máquina no Webots

##Componentes do Grupo

* **Malu Brito**
* **Rodrigo Queiroz**

##Disciplina: MATA64 – Inteligência Artificial
---

##Objetivo do Projeto

Este projeto tem como objetivo equipar o robô terrestre **YouBot**, no simulador **Webots**, para realizar a tarefa de **coletar 15 cubos coloridos** (verde, azul e vermelho) distribuídos aleatoriamente pela arena e **depositá-los nas caixas de cor correspondente**, evitando obstáculos durante toda a navegação.

---

### 1. Requisitos

* Webots (versão compatível com o projeto base)
* Python 3.x

###  2. Instalar Dependências

Antes de executar, é necessário instalar as bibliotecas utilizadas:

```bash
pip install ultralytics
```

```bash
pip install numpy
```

---
* Versão do Python
* Dependências instaladas

---

## Abordagem Utilizada

O sistema de controle do robô combina duas técnicas principais de Inteligência Artificial:

### 1. Rede Neural (YOLO)

Foi utilizado um modelo **YOLO (You Only Look Once)** para:

* Detectar cubos coloridos
* Identificar a cor dos cubos
* Auxiliar na identificação das caixas de destino

O modelo realiza a detecção por meio da **câmera RGB** do robô.

### 2. Lógica Fuzzy

A lógica fuzzy é responsável pelo **controle de navegação do robô**, incluindo:

* Aproximação do alvo
* Correção de alinhamento
* Controle de velocidade
* Desvio de obstáculos

---

## Sensores Utilizados

O robô utiliza exclusivamente:

*  **LIDAR**:

  * Detecção de obstáculos
  * Segmentação de objetos
  * Estimativa de distância

  * **Câmera RGB**:

  * Detecção e classificação de cubos por cor via YOLO
