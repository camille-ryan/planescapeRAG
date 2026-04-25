# TODO

## Frontend

- **Typo reporting UI** — let users flag canonical names or quoted text in the rendered graph that look wrong. Capture `(entity, span, suggested correction)`. Pipe these into the corpus profile's `### OCR corrections` section as new `old → new` bullets, then re-run the affected pipeline stages. The mechanism for applying corrections is already built ([ocr.py](ocr.py) `apply_ocr_corrections`); the gap is the reporting + persistence flow.
