"""Names of the COCO categories in Portuguese, to suggest a mapping for a project that uses them.

A project's classes are free text. When a person writes ``gato`` or ``cão`` for the COCO categories ``cat``
and ``dog``, the mapping of a public model is suggested anyway.
"""

from __future__ import annotations

import unicodedata

# Portuguese names (without accents, lower case, spaces for any separator) → the COCO category in English.
PT_TO_COCO: dict[str, str] = {
    "pessoa": "person",
    "gente": "person",
    "bicicleta": "bicycle",
    "bike": "bicycle",
    "carro": "car",
    "automovel": "car",
    "moto": "motorcycle",
    "motocicleta": "motorcycle",
    "aviao": "airplane",
    "onibus": "bus",
    "trem": "train",
    "caminhao": "truck",
    "barco": "boat",
    "semaforo": "traffic light",
    "hidrante": "fire hydrant",
    "placa de pare": "stop sign",
    "parquimetro": "parking meter",
    "banco": "bench",
    "passaro": "bird",
    "ave": "bird",
    "gato": "cat",
    "cao": "dog",
    "cachorro": "dog",
    "cavalo": "horse",
    "ovelha": "sheep",
    "vaca": "cow",
    "boi": "cow",
    "elefante": "elephant",
    "urso": "bear",
    "zebra": "zebra",
    "girafa": "giraffe",
    "mochila": "backpack",
    "guarda chuva": "umbrella",
    "bolsa": "handbag",
    "gravata": "tie",
    "mala": "suitcase",
    "esqui": "skis",
    "bola": "sports ball",
    "pipa": "kite",
    "taco de beisebol": "baseball bat",
    "luva de beisebol": "baseball glove",
    "skate": "skateboard",
    "prancha de surfe": "surfboard",
    "prancha de surf": "surfboard",
    "raquete de tenis": "tennis racket",
    "garrafa": "bottle",
    "taca de vinho": "wine glass",
    "copo": "cup",
    "xicara": "cup",
    "garfo": "fork",
    "faca": "knife",
    "colher": "spoon",
    "tigela": "bowl",
    "banana": "banana",
    "maca": "apple",
    "sanduiche": "sandwich",
    "laranja": "orange",
    "brocolis": "broccoli",
    "cenoura": "carrot",
    "cachorro quente": "hot dog",
    "rosquinha": "donut",
    "bolo": "cake",
    "cadeira": "chair",
    "sofa": "couch",
    "planta": "potted plant",
    "vaso de planta": "potted plant",
    "cama": "bed",
    "mesa": "dining table",
    "mesa de jantar": "dining table",
    "vaso sanitario": "toilet",
    "televisao": "tv",
    "notebook": "laptop",
    "computador portatil": "laptop",
    "controle remoto": "remote",
    "teclado": "keyboard",
    "celular": "cell phone",
    "telefone celular": "cell phone",
    "micro ondas": "microwave",
    "forno": "oven",
    "torradeira": "toaster",
    "pia": "sink",
    "geladeira": "refrigerator",
    "livro": "book",
    "relogio": "clock",
    "vaso": "vase",
    "tesoura": "scissors",
    "ursinho de pelucia": "teddy bear",
    "secador de cabelo": "hair drier",
    "escova de dentes": "toothbrush",
}


def name_key(name: str) -> str:
    """A name for comparing: lower case, no accents, and spaces for underscores and hyphens.

    Args:
        name: A category or class name.

    Returns:
        The comparable form, for example ``Guarda-chuva`` gives ``guarda chuva``.
    """
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    return " ".join(plain.casefold().replace("_", " ").replace("-", " ").split())


def coco_name(name: str) -> str | None:
    """The COCO category a Portuguese name stands for (plural accepted), or None.

    Args:
        name: A class name written by a person, for example ``gatos``.

    Returns:
        The category in English (``cat``), or None when the name is not a known Portuguese one.
    """
    key = name_key(name)
    if key in PT_TO_COCO:
        return PT_TO_COCO[key]
    for ending in ("s", "es"):
        if key.endswith(ending) and key[: -len(ending)] in PT_TO_COCO:
            return PT_TO_COCO[key[: -len(ending)]]
    return None
