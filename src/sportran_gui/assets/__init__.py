# -*- coding: utf-8 -*-

import json
from pathlib import Path

from importlib import resources

try:
    from importlib import metadata as importlib_metadata
except ImportError:   # pragma: no cover
    import importlib_metadata


def _load_package_metadata():
    try:
        meta = importlib_metadata.metadata('sportran')
        package_version = importlib_metadata.version('sportran')
    except importlib_metadata.PackageNotFoundError:
        meta = {}
        package_version = '0+unknown'

    urls = meta.get_all('Project-URL') if hasattr(meta, 'get_all') else None
    homepage = 'https://github.com/sissaschool/sportran'
    if urls:
        for item in urls:
            if ',' in item:
                label, url = item.split(',', 1)
                if label.strip().lower() == 'homepage':
                    homepage = url.strip()
                    break

    return {
        'version': package_version,
        'author': meta.get('Author', 'Loris Ercole, Riccardo Bertossa, Sebastiano Bisacchi'),
        'author_email': meta.get('Author-email', 'loris.ercole@epfl.ch'),
        'url': meta.get('Home-page', homepage),
        'license': meta.get('License', 'GPL-3.0-or-later'),
        'description': meta.get('Summary', 'SporTran'),
        'classifiers': meta.get_all('Classifier') if hasattr(meta, 'get_all') else [],
        'gui_version': '0.1.2',
        'credits': 'developed at SISSA, Via Bonomea, 265 - 34136 Trieste ITALY',
    }


def _load_text_file(path, fallback):
    if path.is_file():
        return path.read_text(encoding='utf-8')
    return fallback


METADATA = _load_package_metadata()

_dev_states = [x for x in METADATA.get('classifiers', []) if 'Development Status ::' in x]
dev_state = _dev_states[0].split('::')[1].strip() if _dev_states else ''

with resources.as_file(resources.files(__name__).joinpath('languages.json')) as language_path:
    with open(language_path, encoding='utf-8') as stream:
        LANGUAGES = json.load(stream)

with resources.as_file(resources.files(__name__).joinpath('icon.gif')) as icon_path:
    ICON = icon_path.read_bytes()

_ROOT = Path(__file__).resolve().parents[2]
README_MD = _load_text_file(_ROOT / 'README.md', METADATA['description'])
README_GUI_MD = _load_text_file(_ROOT / 'README_GUI.md', METADATA['description'])
