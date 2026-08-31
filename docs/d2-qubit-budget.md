# Dimensão intrínseca, dimensão fractal e a escolha de atributos no QML

Nota de pesquisa. A pergunta **não** é se QSVM ganha de SVM: em dados tabulares clássicos, o SVM costuma ganhar, e isso já está na literatura. A pergunta é **quântico contra quântico**:

> No mesmo modelo QML (QSVM, kNN de fidelidade, VQC, …), é melhor encodar os atributos escolhidos pela dimensão fractal do que encodar todos os atributos, os primeiros componentes de PCA, ou outro seletor clássico?

O que esta linha avalia é a geometria: \(D_2\) (quantos qubits) e o FD-ASE (quais colunas originais). A amostra de linhas é uniforme.

## 1. Dimensão de embedding vs dimensão intrínseca

Um conjunto tabular é uma nuvem de \(N\) pontos em \(\mathbb{R}^{E}\). \(E\) é a **dimensão de embedding** (ou dimensão ambiente): o número de atributos gravados. Quase nunca é o número de graus de liberdade que a nuvem realmente tem.

A **dimensão intrínseca** (ID) é esse número de graus de liberdade: a dimensão do suporte, da variedade, ou do conjunto fractal em que os dados vivem.

Três desenhos mentais:

1. **Círculo em 20-D.** Dois eixos carregam \((\cos t, \sin t)\); os outros 18 são ruído. \(E=20\), mas um único parâmetro \(t\) descreve o sinal. A ID é \(\approx 1\), não 20 e não 2 (o círculo não preenche o disco).
2. **Plano em 100-D.** Dois eixos lineares independentes, o resto constante ou ruído. ID \(= 2\). Aqui PCA e ID coincidem, porque o suporte *é* um subespaço linear.
3. **Swiss roll.** Uma folha 2-D enrolada em 3-D. A ID é 2, mas PCA precisa de **três** componentes para explicar a nuvem no espaço ambiente: o enrolamento é não-linear. Variância linear \(\neq\) dimensão do suporte.

Em QML com *angle encoding*, cada atributo (ou cada PC) vira em geral **um qubit** [havlicek2019supervised, schuld2019quantum]. Encodar \(E\) é encodar a dimensão ambiente. O circuito não “sabe” que 18 eixos são ruído: trata-os como coordenadas reais, e o kernel de fidelidade sente a dimensão **em que os estados são preparados**, não a ID.

Por isso a ID não é um detalhe geométrico. É o orçamento: se o suporte tem ID \(\approx d\), preparar estados em \(q \gg d\) qubits é pedir ao mapa uma dimensão que os dados não têm.

### 1.1 O que a ID não é

- **Não é o número de classes.** Breast cancer tem 2 classes e ID da ordem de 2–4; digits tem 10 classes e um suporte de imagem muito mais rico.
- **Não é o rank da matriz de covariância**, salvo quando o suporte é um subespaço linear. Rank e PCA-95% medem dispersão *linear*.
- **Não precisa ser inteira.** Conjuntos auto-similares (pó de Cantor, algumas misturas de clusters) têm ID fracionária. A dimensão fractal é o instrumento que permite isso.

## 2. Dimensão fractal de correlação \(D_2\)

Há muitas definições de dimensão (Hausdorff, *box-counting* \(D_0\), informação \(D_1\), correlação \(D_2\), …). Para nuvens de pontos em mineração de dados, a que se estima de forma estável é a **dimensão fractal de correlação** \(D_2\) [belussi1995estimating, grassberger1983characterization].

Intuição: conte pares de pontos que caem na mesma célula de uma grade de lado \(r\). Se o suporte se comporta como um conjunto de dimensão \(d\), essa contagem escala como uma potência de \(r\):

\[
S(r) \;=\; \sum_{\text{células } i} C_i(r)^{2} \;\propto\; r^{D_2}.
\]

\(C_i(r)\) é a ocupação da célula \(i\). Em log-log,

\[
D_2 \;=\; \frac{\mathrm{d}\,\log S(r)}{\mathrm{d}\,\log r}.
\]

É a inclinação do gráfico que o LiBOC já desenhava [traina2000fast]. Escalas em que cada ponto está sozinho na sua célula (\(S \approx N\)) não entram no ajuste: aí a grade só vê poeira, não o suporte.

Propriedades que importam para QML:

- **\(D_2 \le E\)**, sempre. Igualdade só se a nuvem preenche o cubo ambiente.
- **Invariante a eixos redundantes.** Se a coluna 7 é um múltiplo da coluna 2, \(D_2\) quase não sobe ao incluir a 7. É exactamente o teste do FD-ASE.
- **Multi-escala.** O ajuste usa vários \(r\). Por isso \(D_2\) não se deixa enganar tão depressa por ruído *local* como um vizinho mais próximo (TwoNN [facco2017estimating] vê a geometria à escala de \(r_1, r_2\); se o ruído de embedding for dessa ordem, inflaciona a ID).
- **Sem rótulo e sem circuito.** Calcula-se classicamente, uma vez, sobre um subconjunto grande (centenas ou milhares de pontos — não sobre os \(n=32\) do kernel NISQ).

O estimador é o box-count LiBOC: normaliza para o cubo unitário, constrói a grade multi-resolução e estima a inclinação na faixa auto-similar.

### 2.1 Dois números, dois papéis: *quantos* e *quais*

\(D_2\) responde **quantos** graus de liberdade o suporte tem. Não escolhe coordenadas.

O **FD-ASE** [sousa2002fractal] (o mesmo algoritmo do pôster SAC 2007) responde **quais** atributos originais bastam para recuperar essa \(D_2\):

1. deita fora constantes (dimensão parcial \(\approx 0\));
2. acrescenta o atributo que mais faz crescer a dimensão parcial \(pD\);
3. pára quando \(pD\) deixa de subir (o novo eixo é correlacionado com os já escolhidos);
4. devolve o conjunto \(E_s\), com \(|E_s| \approx D_2\).

A distinção com PCA é nítida no encoding:

| | PCA | FD-ASE |
|---|---|---|
| O que devolve | misturas lineares de **todos** os eixos | um **subconjunto** das colunas originais |
| Critério | variância | crescimento de \(D_2\) |
| Um qubit encoda | um PC (lixo misturado no sinal) | um atributo que ainda aumenta a ID |
| \(k\) | hiperparâmetro (95% de variância, ou 4–16) | \(\lceil D_2\rceil\) sai da geometria |

PCA **não** é um estimador de ID. É um corte de variância linear. Pode pedir 19 componentes num círculo 1-D embebido em 20-D, porque a standardização transforma ruído em variância unitária. O fractal, no mesmo conjunto, continua a ver \(D_2 \approx 1\).

## 3. Por que isto entra no QML (e por que não se compara com o clássico)

Angle encoding / `ZZFeatureMap`: um qubit por coordenada. O kernel de fidelidade é

\[
K_{ij} \;=\; \bigl|\langle \psi(x_i) \mid \psi(x_j) \rangle\bigr|^{2}.
\]

Em dimensão alta os estados tornam-se quase ortogonais: os off-diagonais concentram-se em 0 e o método deixa de ver geometria [huang2021power, thanasilp2024exponential]. Isso vale para **qualquer** tarefa que beba deste \(K\) — QSVM, kNN quântico, clustering espectral no kernel, one-class, kernel ridge, e o bloco de encoding de um VQC [schuld2021supervised]. Não é um problema de “o classificador clássico é melhor”. É um problema de **preparação de estados**.

A prática NISQ reduz \(E\) até caber no hardware: PCA, autoencoder, seleção manual [belis2025learning]. O \(k\) é um hiperparâmetro validado *depois* no QSVM. Ninguém, que tenhamos encontrado, usa \(D_2\) como orçamento *a priori*, nem FD-ASE como seletor cujo rival é o PCA **dentro do mesmo modelo quântico**.

A tese, numa frase:

> \(\lceil D_2\rceil\) diz quantos qubits o feature map ainda aguenta; o FD-ASE diz quais atributos originais usar. No mesmo QML, isso deve ser melhor (kernel *vivo*, tarefas que dependem de \(K\) não degeneradas) do que encodar todos os atributos, PCA-95%, ou \(k\) eixos aleatórios.

O que **não** é a tese:

- QSVM vs SVM (clássico vs quântico).
- Amplitude encoding (\(\lceil\log_2 E\rceil\) qubits): outra conta; \(D_2\) não encurta o state preparation.
- Projected kernels ou Bit-Flip Tolerance [huang2021power, agliardi2025mitigating]: mudam *como* se avalia \(K\), não *quais* coordenadas se encodam.

## 4. Protocolo: o mesmo modelo quântico, vistas de atributos diferentes

Implementação: [`qml/qubit_budget.py`](../qml/qubit_budget.py), [`qml/features.py`](../qml/features.py) (FD-ASE, PCA, prefixo, aleatório), CLI [`qml/run_qubit_budget.py`](../qml/run_qubit_budget.py).

Comparações, todas sobre o **mesmo** kernel de fidelidade (simulador, extra `qml`):

| Vista | O que o QML vê |
|---|---|
| **FD-ASE** | colunas \(E_s\), \(q=\lvert E_s\rvert \approx \lceil D_2\rceil\) |
| **PCA** | \(q\) primeiros PCs, com \(q\) varrido e também \(q\) de PCA-95% |
| **prefixo / full** | as primeiras (ou todas as) coordenadas, até ao teto de hardware |
| **aleatório** | \(q\) colunas ao acaso (baseline de seleção) |

O juiz não é “acurácia contra SVM”. É se o \(K\) quântico ainda distingue geometria (*kernel alive*: near \(\ge 0{,}25\), razão near/far \(\ge 2\), média off-diagonal \(\ge 0{,}03\)) e as tarefas **quânticas** que partilham \(K\):

| camada | o que correu |
|---|---|
| Feature maps | \(ZZ\) no simulador (varredura) e no IBM; dense-angle e re-uploading **só no simulador** (\(n=32\); breast, moons, pendigits) |
| Kernel | fidelidade \(K_{ij}=\lvert\langle\psi(x_i)\mid\psi(x_j)\rangle\rvert^{2}\) |
| QML sobre \(K\) (simulador, \(n=32\)) | SVM pré-computado (estilo QSVM), kNN de fidelidade, clustering espectral, one-class SVM, KRR |
| Controlo | RBF clássico nas **mesmas** \(q\) coordenadas |
| Hardware | só o kernel \(ZZ\) (\(n=8\)); classificadores não correm no QPU |
| Não usado | VQC treinado, QSVM vs SVM como pergunta |

**Menos qubits ≠ menos pontos.** \(\lceil D_2\rceil\) corta coordenadas no mapa. O \(n=8\) no IBM corta linhas porque cada par é um circuito. A tese é a primeira corte. Não afirmamos que uma amostra menor é melhor.

O RBF no CSV é só diagnóstico: se ele vive e a fidelidade morre, o colapso é do feature map, não da maldição clássica.

\(D_2\), TwoNN e PCA-95% estimam-se num subconjunto clássico grande (até 4000 pontos), não nos \(n=32\) do kernel.

```bash
python d2-qubit-budget/qml/run_qubit_budget.py
python d2-qubit-budget/qml/run_qubit_budget.py --fast
```

## 5. O que os números mostram (simulador, \(n=32\))

Leitura **quântico–quântico**. A coluna *vivo* é o último \(q\) da curva PCA/fidelidade. PCA-95% é o \(q\) que a prática QML escolheria. \(\lceil D_2\rceil\) é o \(q\) que o fractal escolhe.

| conjunto | \(E\) | \(D_2\) | \(\lceil D_2\rceil\) | PCA-95% | vivo (fidelidade) |
|---|---:|---:|---:|---:|---:|
| intrinsic2 | 20 | 0,72 | 2 | 19 | 3 |
| moons | 20 | 0,72 | 2 | 19 | 3 |
| iris | 4 | 1,92 | 2 | 2 | 3 |
| breast | 30 | 2,49 | 3 | 10 | **3** |
| diabetes | 10 | 5,82 | 6 | 8 | 3 |
| wine | 13 | 6,48 | 7 | 10 | 2 |
| digits | 64 | 0,91 | 2 | 40 | 4 |
| pendigits | 16 | 5,35 | 6 | 10 | 5 |
| onebig | 8 | 2,27 | 3 | 8 | 7 |

- **Encodar “como o PCA da prática QML” mata o kernel** onde o fractal o mantém vivo. Breast: PCA-95% pede 10 qubits; a fidelidade já colapsou em \(q=4\). Moons/intrinsic2: PCA pede 19; o mapa de ângulo vive em 2–3.
- **Encodar todos os atributos** (ou o prefixo até 8) é o mesmo erro quando \(E\) é grande: é a dimensão ambiente, não a ID.
- **Pendigits** (\(E=16\), \(D_2\approx 5{,}4\)): o joelho da fidelidade está em 5, junto do teto fractal 6, não em 10 (PCA) nem em 16 (tudo).
- **Eixos aleatórios** não substituem o FD-ASE: no breast a curva aleatória/fidelidade não sustenta o critério *alive*.
- Falhas honestas: OneBig (clusters triviais, o kernel vive além de \(\lceil D_2\rceil\)); wine/diabetes (ID alta, \(n=32\) concentra *antes* do teto); digits (\(E=64\), box-count no chão — não confiar em LiBOC quando \(E \gg \log N\)).

![Tetos rivais vs último q vivo](figures/qubit-budget/qubit_budget_ceilings.png)

*Leitura da figura.* Um ponto na diagonal \(y=x\) significa que o teto proposto coincide com o último \(q\) em que o kernel de fidelidade ainda está vivo. À esquerda, \(\lceil D_2\rceil\) acompanha esse joelho (breast em \(3\); pendigits \(6\) vs \(5\)). Ao centro, TwoNN inflaciona moons/intrinsic2. À direita, PCA-95% pede \(10\)–\(40\) qubits quando o kernel já morreu em \(q\le 5\) — esse painel tem eixo \(x\) próprio para os joelhos vivos (\(2\)–\(7\)) não ficarem esmagados.

### 5.1 Breast: o mesmo QSVM, \(q\) fractal vs \(q\) PCA

\(E=30\), \(D_2\approx 2{,}49\), \(\lceil D_2\rceil=3\), PCA-95%=10. Kernel de fidelidade, vista PCA (esta amostra \(n=32\)):

| q | vivo? | mean \(K\) | near / far |
|---|---|---|---|
| 2 | não (razão \(<2\)) | 0,30 | 0,49 / 0,34 |
| **3 = ⌈D2⌉** | **sim** | 0,14 | 0,27 / 0,12 |
| 4 | não | 0,07 | 0,11 / 0,07 |
| 8 | não | 0,00 | 0,00 / 0,01 |

O F1 do QSVM com \(n=32\) saltita; a concentração não. Encodar 10 PCs, neste mapa, é encodar um kernel já diagonal.

![Breast](figures/qubit-budget/qubit_budget_breast_cancer.png)

*Leitura da figura.* Linha a tracejado vertical cinzenta: PCA-95% (\(q=10\)). Pontilhado: teto fractal (\(q=3\)). Sólido = kernel de fidelidade (o objecto QML); tracejado = RBF clássico nas mesmas coordenadas. A média off-diagonal da fidelidade cai para \(0\) quando \(q\) cresce; o RBF não — o colapso é do mapa de ângulo. No `pca/fidelity`, kNN e F1 do SVM têm o pico em \(q=3\) e pioram depois. Encodar \(3\) PCs (largura fractal) deixa o QSVM/kNN com um \(K\) vivo; encodar \(10\), como a prática PCA, alimenta os mesmos algoritmos com um kernel já diagonal.

Figuras dos outros conjuntos: `docs/figures/qubit-budget/` (eixos em inglês).

## 6. Mais do que um atributo por qubit

$\lceil D_2\rceil$ orça o *espaço de Hilbert* (quantos qubits), não a regra «um atributo = um qubit». Dense-angle ($RY$+$RZ$ por qubit) e *data re-uploading* [perezsalinas2020data] mantêm o kernel vivo no $q$ fractal ao empilhar duas coordenadas PCA por qubit. Empilhar até ao corte PCA-95% *nesse mesmo* $q$ (12 PCs em 3 qubits no breast; 16 PCs em 2 qubits nos moons) mata ou inverte a geometria near/far.

**Veredicto para encodings:** no simulador, sim — trabalhar na largura fractal é melhor do que encodar a largura PCA-95% (ou todo o \(E\)); não se recupera essa largura extra empilhando-a em \(\lceil D_2\rceil\) qubits. Dense-angle e re-uploading **foram** testados (simulador, \(n=32\)). No IBM correu só o mapa \(ZZ\). CSV e figura: [`docs/encodings/`](encodings/).

## 7. Hardware (`ibm_fez`, 30 de agosto de 2026)

Seis jobs Sampler, 256 shots, $n=8$ (28 circuitos compute–uncompute), entanglement linear, `reps=1`. Matrizes em [`docs/hardware/`](hardware/).

| job | $q$ | MAE vs SV | mean $K$ HW | mean $K$ SV |
|---|---:|---:|---:|---:|
| blobs | 5 | 0,077 | 0,234 | 0,315 |
| blobs | 7 | 0,054 | 0,142 | 0,202 |
| breast PCA | **3 = ⌈D2⌉** | **0,021** | **0,082** | **0,081** |
| breast PCA | 7 | 0,003 | 0,004 | 0,003 |
| breast FD-ASE (cols. 27, 21, 9, 8) | 4 | **0,012** | 0,065 | 0,068 |
| breast aleatório (cols. 15, 18, 23) | 3 | 0,030 | 0,164 | 0,166 |

No teto PCA e no FD-ASE o hardware reproduz o kernel exacto (MAE $0{,}012$–$0{,}021$). Em $q=7$ o simulador já colapsou e o dispositivo concorda. O controlo aleatório não colapsa, mas $n=8$ não chega para o critério near/far (o PCA $q=3$ nesta subamostra também falha *alive* — essa regra foi calibrada em $n=32$).

![MAE e concentração no hardware](hardware/figures/hardware_mae_concentration.png)

*Leitura.* Esquerda: o dispositivo implementa o mapa (MAE baixo = fiel ao statevector). Direita: em \(q=7\) as barras HW e SV estão no chão — o colapso já está no \(K\) exacto. FD-ASE (\(q=4\)) é o mapa mais fiel (MAE \(0{,}012\)). O aleatório não colapsa, mas \(n=8\) não decide near/far.

## 8. Outras técnicas QML (sempre o mesmo \(K\))

Tudo o que prepara \(\lvert\psi(x)\rangle\) por ângulos herda o teto e o seletor: VQC, kNN quântico, distâncias em embeddings, reservoirs. Amplitude encoding e hamiltonianos físicos cujo grau de liberdade não é um vetor tabular **não** herdam.

## 9. Limites

- \(D_2\) por box-count precisa de amostra e de várias escalas; dezenas de pontos não chegam.
- FD-ASE escolhe colunas originais, não misturas; se o sinal *só* existir numa direcção oblíqua, PCA pode ser o seletor certo — e o protocolo tem de mostrar isso quando acontecer.
- O critério *alive* é operacional, não um teste de hipótese.
- Simulador, entanglement linear, `reps=1`. O joelho deve *acompanhar* $D_2$, não copiar o valor 3 em todo circuito.
- Hardware: $n=8$, um backend, 256 shots, sem mitigação de erro.

## 10. Conclusão

**Menos qubits, sim.** No mesmo modelo quântico — mapa \(ZZ\) (no simulador e no IBM) ou dense-angle / re-uploading (**só simulador**), kernel de fidelidade, e depois QSVM, kNN de fidelidade, clustering espectral, one-class SVM ou KRR sobre esse \(K\) — encodar em \(q=\lceil D_2\rceil\) mantém o kernel vivo. Encodar a largura PCA-95% ou todos os \(E\) atributos entrega a esses algoritmos um kernel já colapsado (breast: vivo em \(3\), morto em \(4\)–\(10\); moons: vivo em \(2\)–\(3\), o PCA pediria \(19\)). O FD-ASE nomeia as colunas originais dessa largura. No simulador, dois atributos por qubit ainda funcionam no \(q\) fractal; despejar o PCA-95% nesses qubits não.

**Menos pontos, não — isso não foi o teste.** \(D_2\) estima-se em até \(4000\) linhas clássicas. O kernel QML usa \(n=32\) no simulador e \(n=8\) no IBM só porque cada par é um circuito. Não afirmamos que uma amostra menor é melhor. No dispositivo, \(n=8\) chega para ver que \(q=3\) é fiel e \(q=7\) já é diagonal; não chega para ordenar FD-ASE contra colunas aleatórias.

Rascunho LaTeX (formato tipo PKDD/LNCS): [`paper/d2-qubit-budget.tex`](paper/d2-qubit-budget.tex).

## 11. Reprodução

```bash
python3 -m pip install -e ".[dev,qml]"
python3 d2-qubit-budget/qml/run_qubit_budget.py
python3 d2-qubit-budget/qml/run_packed_encoding.py
python3 d2-qubit-budget/qml/run_hardware_article.py   # IBM; grava docs/hardware/
python3 -m pytest -q d2-qubit-budget/tests
```

Referências: [`referencias.bib`](../../docs/referencias.bib). Versão em inglês: [`d2-qubit-budget.en.md`](d2-qubit-budget.en.md).
