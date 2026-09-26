<div align="center">

<img src="docs/logo.svg" width="112" alt="RLCore">

# RLCore

**Каркас для обучения с подкреплением.**

<a href="README.md"><img src="https://img.shields.io/badge/English-5b8cff?style=for-the-badge&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGlkPSJmbGFnLWljb25zLWdiIiB2aWV3Qm94PSIwIDAgNjQwIDQ4MCI+CiAgPHBhdGggZmlsbD0iIzAxMjE2OSIgZD0iTTAgMGg2NDB2NDgwSDB6Ii8+CiAgPHBhdGggZmlsbD0iI0ZGRiIgZD0ibTc1IDAgMjQ0IDE4MUw1NjIgMGg3OHY2Mkw0MDAgMjQxbDI0MCAxNzh2NjFoLTgwTDMyMCAzMDEgODEgNDgwSDB2LTYwbDIzOS0xNzhMMCA2NFYweiIvPgogIDxwYXRoIGZpbGw9IiNDODEwMkUiIGQ9Im00MjQgMjgxIDIxNiAxNTl2NDBMMzY5IDI4MXptLTE4NCAyMCA2IDM1TDU0IDQ4MEgwek02NDAgMHYzTDM5MSAxOTFsMi00NEw1OTAgMHpNMCAwbDIzOSAxNzZoLTYwTDAgNDJ6Ii8+CiAgPHBhdGggZmlsbD0iI0ZGRiIgZD0iTTI0MSAwdjQ4MGgxNjBWMHpNMCAxNjB2MTYwaDY0MFYxNjB6Ii8+CiAgPHBhdGggZmlsbD0iI0M4MTAyRSIgZD0iTTAgMTkzdjk2aDY0MHYtOTZ6TTI3MyAwdjQ4MGg5NlYweiIvPgo8L3N2Zz4K" alt="English"></a>
<a href="README.ru.md"><img src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-30363d?style=for-the-badge&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGlkPSJmbGFnLWljb25zLXJ1IiB2aWV3Qm94PSIwIDAgNjQwIDQ4MCI+CiAgPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTAgMGg2NDB2MTYwSDB6Ii8+CiAgPHBhdGggZmlsbD0iIzAwMzlhNiIgZD0iTTAgMTYwaDY0MHYxNjBIMHoiLz4KICA8cGF0aCBmaWxsPSIjZDUyYjFlIiBkPSJNMCAzMjBoNjQwdjE2MEgweiIvPgo8L3N2Zz4K" alt="Русский"></a>

[![Python](https://img.shields.io/badge/Python-3.10%2B-5b8cff)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-3ddbc4)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-31%20passed-3ddbc4)](tests)
[![License](https://img.shields.io/badge/license-MIT-5b8cff)](LICENSE)
[![Stars](https://img.shields.io/github/stars/himsson/RLCore?style=flat&color=f5a524)](https://github.com/himsson/RLCore/stargazers)

</div>

---

RLCore — это основа для обучения с подкреплением. Алгоритмов здесь нет: есть всё, что их окружает.

Векторизованные среды (синхронные и в процессах), replay-буфер с n-step, приоритетный на SumTree,
rollout-буфер с GAE, нормализация наблюдений и наград, цикл обучения с оценкой и чекпоинтами,
resume вплоть до состояния генератора случайных чисел, распределения с маской действий, сети,
логи в jsonl, csv и tensorboard.

Свой алгоритм — это один файл в `algos/`: три метода и строчка `@ALGOS.register("dqn")`.
Своя среда — один класс в `envs/`. Подключать ничего не нужно, пакеты находят себя сами.

Отдельно стоит сказать, что в базе уже правильно сделаны вещи, на которых обычно и горит обучение:
`terminated` и `truncated` разделены по всему стеку, GAE бутстрапит на обрыве по таймеру, n-step
останавливается на настоящем терминале, а реальное последнее наблюдение переживает авторесет.
Это те баги, которые не падают с ошибкой, а просто молча дают агента похуже.

## Быстрый старт

Нужен Python 3.10+, `numpy`, `pyyaml`, `torch`.

```bash
python train.py --config configs/default.yaml
```

## Поддержать

Если RLCore сэкономил тебе день отладки — ⭐ на репозитории помогает проекту быть заметнее.

## 📄 Лицензия

MIT © 2026 himsson. Подробности в [LICENSE](LICENSE).
