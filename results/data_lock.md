# Data lock (computed from raw UCI files)

| UCI | Dataset | Niche | Rows (UCI meta) | Rows (file) | Features | Pos / Neg | Majority | Dup rows | Unique | Conflict groups | After dedup pos/neg |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 863 | Maternal Health Risk (rural Bangladesh) | D | 1013 | 1014 | 6 | 608 / 406 | 0.600 | 568 | 446 | 30 | 212 / 234 |
| 193 | Cardiotocography | D | 2126 | 2126 | 21 | 471 / 1655 | 0.778 | 13 | 2113 | 2 | 467 / 1646 |
| 508 | QSAR oral toxicity | B | 8992 | 8992 | 1024 | 741 / 8251 | 0.918 | 477 | 8515 | 11 | 719 / 7796 |
| 509 | QSAR androgen receptor | B | 1687 | 1687 | 1024 | 199 / 1488 | 0.882 | 0 | 1687 | 1 | 199 / 1488 |
| 254 | QSAR biodegradation | B | 1055 | 1055 | 41 | 356 / 699 | 0.663 | 3 | 1052 | 0 | 354 / 698 |

863 demotion rule (>= 400 unique rows and >= 100 per class after dedup): PASSES, stays core (rows 446, positives 212, negatives 234).

579 (Myocardial infarction complications) is not loaded yet: its post-admission drop list still has to be built from the UCI variable table.
