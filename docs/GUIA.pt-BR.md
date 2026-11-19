# ActiveLabel — guia em português

Conjuntos de imagens da curadoria ao modelo treinado, por etapas, numa interface web
simples (português e inglês, tema claro e escuro).

**Etapa 1 (esta versão):** criar um projeto, escolher a tarefa, enviar imagens, carregar
anotações que já existam (conferidas com a tarefa) e fazer a curadoria das imagens:
achar imagens corrompidas, borradas, escuras, claras demais, com baixo contraste,
pequenas ou quase duplicadas, e excluí-las. As próximas etapas (anotação, vários
treinos, avaliação, revisão das predições como novo ground truth) partem daqui.

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

## Como funciona

1. **Novo projeto**: nome, descrição e tarefa (classificação, detecção, segmentação ou
   pontos-chave). O número do projeto é automático. As classes são opcionais; podem vir
   dos arquivos de anotação.
2. **Imagens**: solte arquivos, uma pasta ou um .zip. Os arquivos são guardados sem
   alteração; imagens idênticas entram uma só vez; arquivos ilegíveis aparecem na lista
   com o motivo. Arquivos de anotação enviados junto com as imagens são indicados para a
   etapa 2.
3. **Anotações** (opcional): solte os arquivos de anotação (ou um .zip). O formato é
   detectado (COCO JSON, YOLO txt com `data.yaml`/`classes.txt`, Pascal VOC XML, XML do
   CVAT para imagens, JSON do Label Studio), os arquivos são casados com as imagens do
   projeto e um relatório mostra o que foi encontrado e os problemas. Nada é importado
   até você escolher: *Importar*, *Trocar a tarefa do projeto e importar* (quando as
   anotações são de outra tarefa), *Converter e importar* (por exemplo contornos em
   caixas) ou *Descartar*.
4. **Curadoria**: *Analisar imagens* mede brilho, contraste, nitidez, tamanho e um hash
   perceptual de cada imagem (uma tarefa com progresso, que pode ser cancelada). Os
   filtros por problema mostram as imagens apontadas; selecione e *Exclua* (reversível,
   aba *Excluídas*) ou apague de vez (pede confirmação). *Abrir no FiftyOne* abre o mesmo
   filtro no App do FiftyOne; imagens marcadas com a tag `exclude` lá voltam com
   *Trazer exclusões do FiftyOne*.

As tarefas demoradas aparecem na barra **Tarefas**, no pé da página do projeto, com
progresso, *Cancelar* e *Ver resultado*.

## Configuração

| Variável | Padrão | O que faz |
| --- | --- | --- |
| `ACTIVELABEL_WORKSPACE` | `~/.activelabel` | Onde ficam projetos, imagens e anotações |
| `ACTIVELABEL_LANG` | `en` | Idioma padrão da interface (`en` ou `pt`) |
| `ACTIVELABEL_IMPORT_ROOT` | (desligado) | Pasta desta máquina cujas subpastas podem ser importadas pelo caminho |
