# ActiveLabel — guia em português

Conjuntos de imagens da curadoria ao modelo treinado, por etapas, numa interface web
simples (português e inglês, tema claro e escuro).

Um projeto vai das imagens ao modelo treinado em seis passos, em dois grupos: **Dados**
(imagens, anotações, curadoria) e **Modelo** (divisão dos dados, treinamento, modelos). Um
modelo treinado pode então pré-anotar imagens novas, que uma pessoa revisa no CVAT ou no
FiftyOne e devolve ao próximo treinamento.

## Instalar

Python 3.10 ou mais novo.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Rodar

```bash
activelabel serve --open
```

Ou, sem ativar o ambiente: `activelabel.cmd serve --open` (Windows) ou
`./activelabel.sh serve --open` (Linux, macOS). A interface abre em
`http://127.0.0.1:8090`. O serviço não tem login: mantenha o host padrão `127.0.0.1`,
a não ser numa rede confiável.

## Passo a passo

### Criar o projeto

Na lista de projetos, *Novo projeto*: nome, descrição e tarefa (classificação, detecção,
segmentação ou pontos-chave). O número do projeto é automático. As classes são opcionais;
podem vir dos arquivos de anotação. Um projeto que já existe no CVAT ou no Label Studio pode
ser importado em vez de criado (veja [IMPORT.md](IMPORT.md), em inglês). A página do projeto
mostra uma barra compacta com os seis passos, em dois grupos.

### Grupo Dados

**1. Imagens.** Duas abas:

- *Galeria*: a grade de imagens, com filtros por task e por job.
- *Tasks e jobs*: as tasks do projeto (cada envio de imagens, ou cada vídeo, é uma task),
  divididas em jobs do tamanho que você escolher, como no CVAT. Em *Nova task* solte
  arquivos, uma pasta, um .zip ou um vídeo. Os arquivos são guardados sem alteração; imagens
  idênticas entram uma só vez; arquivos ilegíveis aparecem na lista com o motivo. Detalhes:
  [TASKS.md](TASKS.md) e [VIDEO.md](VIDEO.md).

**2. Anotações** (opcional). Cinco abas:

- *Resumo*: o que o projeto já tem (anotadas, a revisar, classes).
- *Pré-anotar*: um modelo salvo sugere anotações para as imagens de uma task (veja o fim
  deste guia e [PREANNOTATE.md](PREANNOTATE.md)).
- *CVAT*: envia imagens ao CVAT e sincroniza as anotações nos dois sentidos, com uma revisão
  antes de qualquer mudança ([ANNOTATION.md](ANNOTATION.md) e [SYNC.md](SYNC.md)). Se o
  projeto vinculado foi apagado no CVAT, a tela avisa; use *Desvincular* e compare de novo
  para criar um projeto novo.
- *FiftyOne*: edite as anotações no App do FiftyOne e traga de volta.
- *Importar anotações*: solte os arquivos de anotação (ou um .zip). O formato é detectado
  (COCO JSON, YOLO txt com `data.yaml`/`classes.txt`, Pascal VOC XML, XML do CVAT para
  imagens ou vídeo, MOT, JSON do Label Studio, pastas de classes), os arquivos são casados
  com as imagens e um relatório mostra o que foi encontrado e os problemas. Nada é importado
  até você escolher: *Importar*, *Trocar a tarefa do projeto e importar* (quando as anotações
  são de outra tarefa), *Converter e importar* (por exemplo contornos em caixas) ou
  *Descartar*. Anotações feitas para um vídeo inteiro vão para uma task de vídeo, casadas
  pelo número do quadro ([VIDEO.md](VIDEO.md)).

**3. Curadoria.** *Analisar imagens* mede brilho, contraste, nitidez, tamanho e um hash
perceptual de cada imagem. Os filtros por problema (corrompida, borrada, escura, clara demais,
baixo contraste, pequena, quase duplicada) mostram as imagens apontadas; selecione e
*Exclua* (reversível, aba *Excluídas*) ou apague de vez (pede confirmação). *Abrir no
FiftyOne* abre o mesmo filtro no App do FiftyOne; imagens marcadas com a tag `exclude` lá
voltam com *Trazer exclusões do FiftyOne*.

### Grupo Modelo

**4. Divisão dos dados.** Cria uma *versão do dataset*: partes de treino, validação e teste
(ou K dobras mais uma parte de teste). Quadros do mesmo vídeo e quase duplicadas ficam juntos,
e as classes são distribuídas de forma equilibrada. Escolha as tasks, as porcentagens (números
que somam 100; para treinar é preciso ter validação), imagens de fundo, modo de vídeos e a
semente; a prévia mostra os avisos antes de salvar. Detalhes: [DATASETS.md](DATASETS.md).

**5. Treinamento.** Escolha a versão do dataset, o modelo do catálogo e como começar
(transferência de aprendizado, ajuste fino ou do zero), em execução única, busca em grade ou
todas as dobras. Há curvas ao vivo e avaliação na parte de teste. Estão disponíveis
classificação, detecção, segmentação semântica e pontos-chave (torchvision); os modelos
YOLO/Ultralytics estão planejados, ainda não disponíveis. Os pacotes necessários são
instalados sob demanda, num ambiente separado. Detalhes: [TRAINING.md](TRAINING.md).

**6. Modelos.** Salve uma execução concluída como modelo, compare os modelos, escolha o
modelo atual, baixe os pesos e exporte para ONNX (classificação e segmentação semântica).

### Pré-anotar imagens novas com um modelo

1. Em *Imagens*, aba *Tasks e jobs*, crie uma *Nova task* com as imagens novas.
2. Em *Anotações*, aba *Pré-anotar*: escolha o modelo (o atual, por padrão), a task, quais
   imagens (sem anotação, ou pré-anotadas e ainda não revisadas) e a confiança mínima (de 0,05
   a 0,95; a tela propõe 0,5 para classificação e 0,3 para detecção e pontos-chave). Inicie.
   Imagens ilegíveis são contadas e puladas.
3. Em *Para revisar*, use *Revisar no CVAT* ou *Revisar no FiftyOne*, corrija o que for
   preciso e traga de volta (sincronização com o CVAT, ou sessão de edição do FiftyOne). Ou
   escolha *Aceitar como estão*.
4. Crie uma nova versão em *Divisão dos dados* e treine de novo. Pré-anotações não revisadas
   nunca entram numa versão do dataset.

### Atividades

As tarefas demoradas (análise, envio, sincronização, treinamento, exportação, pré-anotação)
aparecem num cartão pequeno no canto inferior direito da página do projeto, com progresso,
*Cancelar* e *Ver resultado*. Ele pode ser minimizado ou fechado; as atividades concluídas
ficam no histórico. Só um processo pesado (treinamento, exportação ou pré-anotação) roda de
cada vez; os outros esperam na fila.

Mais: [DESIGN.md](DESIGN.md) (dados em disco e API HTTP, em inglês) e
[FUTURE-LLM.md](FUTURE-LLM.md) (uso planejado de modelos de linguagem, em inglês).

## Configuração

| Variável | Padrão | O que faz |
| --- | --- | --- |
| `ACTIVELABEL_WORKSPACE` | `~/.activelabel` | Onde ficam projetos, imagens e anotações |
| `ACTIVELABEL_LANG` | `en` | Idioma padrão da interface (`en` ou `pt`) |
| `ACTIVELABEL_IMPORT_ROOT` | (desligado) | Pasta desta máquina cujas subpastas podem ser importadas pelo caminho |
