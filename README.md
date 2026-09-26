<div align="center">

<img src="docs/logo.svg" width="112" alt="RLCore">

# RLCore

**A reinforcement learning framework.**

<a href="README.md"><img src="https://img.shields.io/badge/English-5b8cff?style=for-the-badge&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGlkPSJmbGFnLWljb25zLWdiIiB2aWV3Qm94PSIwIDAgNjQwIDQ4MCI+CiAgPHBhdGggZmlsbD0iIzAxMjE2OSIgZD0iTTAgMGg2NDB2NDgwSDB6Ii8+CiAgPHBhdGggZmlsbD0iI0ZGRiIgZD0ibTc1IDAgMjQ0IDE4MUw1NjIgMGg3OHY2Mkw0MDAgMjQxbDI0MCAxNzh2NjFoLTgwTDMyMCAzMDEgODEgNDgwSDB2LTYwbDIzOS0xNzhMMCA2NFYweiIvPgogIDxwYXRoIGZpbGw9IiNDODEwMkUiIGQ9Im00MjQgMjgxIDIxNiAxNTl2NDBMMzY5IDI4MXptLTE4NCAyMCA2IDM1TDU0IDQ4MEgwek02NDAgMHYzTDM5MSAxOTFsMi00NEw1OTAgMHpNMCAwbDIzOSAxNzZoLTYwTDAgNDJ6Ii8+CiAgPHBhdGggZmlsbD0iI0ZGRiIgZD0iTTI0MSAwdjQ4MGgxNjBWMHpNMCAxNjB2MTYwaDY0MFYxNjB6Ii8+CiAgPHBhdGggZmlsbD0iI0M4MTAyRSIgZD0iTTAgMTkzdjk2aDY0MHYtOTZ6TTI3MyAwdjQ4MGg5NlYweiIvPgo8L3N2Zz4K" alt="English"></a>
<a href="README.ru.md"><img src="https://img.shields.io/badge/%D0%A0%D1%83%D1%81%D1%81%D0%BA%D0%B8%D0%B9-30363d?style=for-the-badge&logo=data:image/svg%2Bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGlkPSJmbGFnLWljb25zLXJ1IiB2aWV3Qm94PSIwIDAgNjQwIDQ4MCI+CiAgPHBhdGggZmlsbD0iI2ZmZiIgZD0iTTAgMGg2NDB2MTYwSDB6Ii8+CiAgPHBhdGggZmlsbD0iIzAwMzlhNiIgZD0iTTAgMTYwaDY0MHYxNjBIMHoiLz4KICA8cGF0aCBmaWxsPSIjZDUyYjFlIiBkPSJNMCAzMjBoNjQwdjE2MEgweiIvPgo8L3N2Zz4K" alt="Русский"></a>

[![Python](https://img.shields.io/badge/Python-3.10%2B-5b8cff)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-3ddbc4)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-31%20passed-3ddbc4)](tests)
[![License](https://img.shields.io/badge/license-MIT-5b8cff)](LICENSE)
[![Stars](https://img.shields.io/github/stars/himsson/RLCore?style=flat&color=f5a524)](https://github.com/himsson/RLCore/stargazers)

</div>

---

RLCore is a foundation for reinforcement learning. There are no algorithms here — everything
around them is.

Vectorized environments (sync and subprocess), a replay buffer with n-step, a prioritized one on a
SumTree, a rollout buffer with GAE, observation and reward normalization, a training loop with
evaluation and checkpoints, resume down to the RNG state, distributions with action masking,
networks, and logs in jsonl, csv and tensorboard.

Your algorithm is one file in `algos/`: three methods and a single `@ALGOS.register("dqn")` line.
Your environment is one class in `envs/`. Nothing to wire up — packages find themselves.

The things that usually break training are already right here: `terminated` and `truncated` are
separated throughout, GAE bootstraps on time-limit cutoffs, n-step stops at real terminals, and the
true final observation survives autoreset. Those are the bugs that never raise an error — they just
quietly give you a worse agent.

## Quick start

Needs Python 3.10+, `numpy`, `pyyaml`, `torch`.

```bash
python train.py --config configs/default.yaml
```

## Support

If RLCore saved you a day of debugging — a ⭐ on the repository helps it get noticed.

## 📄 License

MIT © 2026 himsson. See [LICENSE](LICENSE).
