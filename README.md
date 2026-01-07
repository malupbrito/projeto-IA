# Projeto YouBot – Aprendizagem de máquina no Webots

**Disciplina:** MATA64 – Inteligência Artificial
**Semestre:** 2025.2

---

##  Componentes do Grupo

* **Malu Brito**
* **Rodrigo Queiroz**

---

##  Objetivo do Projeto

Este projeto tem como objetivo equipar o robô terrestre **YouBot**, no simulador **Webots**, para coletar **15 cubos coloridos** (verde, azul e vermelho) distribuídos aleatoriamente pela arena e depositá-los nas **caixas de cor correspondente**, evitando obstáculos durante toda a navegação.

---

## 📌 Requisitos

* Webots (versão compatível com o projeto base fornecido)
* Python 3.x

---

##  Instalação de Dependências

Antes de executar o projeto, instale as bibliotecas necessárias:

```bash
pip install ultralytics
pip install numpy
```

---

##  Abordagem Utilizada

O sistema de controle do robô combina duas técnicas principais de Inteligência Artificial:

### 1. Rede Neural – YOLO

Foi utilizado um modelo **YOLO (You Only Look Once)** para:

* Detectar cubos na imagem da câmera
* Identificar a **cor dos cubos** (verde, azul ou vermelho)
* Auxiliar na identificação das caixas de destino

A detecção é feita a partir da **câmera RGB** do robô.

### 2. Lógica Fuzzy

A lógica fuzzy é responsável pelo controle de navegação do robô, incluindo:

* Aproximação do alvo
* Correção de alinhamento
* Controle de velocidade
* Desvio de obstáculos

As decisões de movimento são obtidas por meio de **regras fuzzy** e defuzzificação por média ponderada.

---

##  Sensores Utilizados

O robô utiliza exclusivamente os seguintes sensores:

### 🔹 LIDAR

* Detecção de obstáculos
* Segmentação de objetos
* Estimativa de distância

### 🔹 Câmera RGB

* Detecção e classificação de cubos por cor via YOLO


