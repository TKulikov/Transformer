# Механика выигрыша memctx (исходный порядок дерева)

Кластеров: std=30, mem=29.

| Задача | gained (mem нашёл, std нет) | lost | cos ключа std↔mem | кластер std→mem |
|---|---|---|---|---|
| retry_backoff | retry | — | retry: cos=0.9987 cl4→4 |
| parse_clamp | — | — | — |
| batch_dedupe | — | — | — |
| slug_total | — | — | — |
| invoice_summary | — | — | — |

cos ключа — насколько контекстуализация живой памятью сдвинула эмбеддинг функции;
кластер std→mem — сменила ли функция группу. Данные: detail.json.
