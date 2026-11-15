# Interlocutor: удержание ГЛОБАЛЬНОЙ ТЕМЫ диалога без полного контекста

window=2, 3 диалога. Метрика: topic_continuation
(cos ответа на фиксированный эталон глобальной темы, сравнимо между условиями).
Исходное назначение механизма слияний — собеседник, держащий направление.

| Условие | topic_continuation | tone | по диалогам |
|---|---|---|---|
| no_memory | 0.7541 | 0.8352 | trip=0.7564, guitar=0.7479, garden=0.758 |
| rag_plain | 0.7604 | 0.8479 | trip=0.7691, guitar=0.7491, garden=0.7629 |
| rag_gist | 0.7563 | 0.8436 | trip=0.7639, guitar=0.7523, garden=0.7526 |
| centered_rag | 0.7594 | 0.8429 | trip=0.7642, guitar=0.7481, garden=0.7659 |
| full_context | 0.7496 | 0.8371 | trip=0.7509, guitar=0.7434, garden=0.7544 |

rag_gist vs rag_plain = вклад живого merge-gist в удержание глобальной темы.
centered_rag = v3 альтернатива; full_context = upper bound. Данные: `<dlg>_<cond>.json`.
