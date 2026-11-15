# Файл-резидуальный поиск (taskq_large): разреживание внутри файла

Ключ = латентный эмбеддинг функции в контексте класса − центроид файла;
запрос: stage 1 — файл по центроиду, stage 2 — (q − центроид) против residual-ключей.
k=6 заметок. Ингестия (с): {'fileres': 41.8, 'flat_legacy': 63.0, 'hier': 73.1}.

| Условие | Reuse /10 | Retrieval hits /10 | Valid /5 |
|---|---|---|---|
| fileres_f1 | 6 | 7 | 1 |
| fileres_f2 | 6 | 6 | 1 |
| fileres_f3 | 6 | 6 | 0 |
| fileres_fused_f2 | 5 | 7 | 0 |
| flat_legacy | 8 | 5 | 1 |
| hier_flat | 7 | 6 | 0 |

## Выбранные файлы (stage 1)

| Условие | retry_backoff | parse_clamp | batch_dedupe | slug_total | invoice_summary |
|---|---|---|---|---|---|
| fileres_f1 | services/batch_service.py | services/visit_service.py | services/batch_service.py | services/document_service.py | handlers/visit_handler.py |
| fileres_f2 | services/batch_service.py, services/visit_service.py | services/visit_service.py, services/session_service.py | services/batch_service.py, services/session_service.py | services/document_service.py, services/visit_service.py | handlers/visit_handler.py, handlers/batch_handler.py |
| fileres_f3 | services/batch_service.py, services/visit_service.py, services/route_service.py | services/visit_service.py, services/session_service.py, services/batch_service.py | services/batch_service.py, services/session_service.py, services/visit_service.py | services/document_service.py, services/visit_service.py, services/subscription_service.py | handlers/visit_handler.py, handlers/batch_handler.py, models/subscription.py |
| fileres_fused_f2 | services/batch_service.py, services/subscription_service.py | services/subscription_service.py, services/visit_service.py | services/batch_service.py, services/visit_service.py | services/shipment_service.py, services/subscription_service.py | services/visit_service.py, services/document_service.py |
| flat_legacy | — | — | — | — | — |
| hier_flat | — | — | — | — | — |

fileres_f1/f2/f3 — beta=0, k_files=1/2/3; fileres_fused_f2 — beta=1 (fused-ключ);
flat_legacy / hier_flat — референсы из search_retest. Данные: `<cond>.json`.
