# W7 parse report — Bruneau, Pluta & Saxe (2011) stimuli

Source: `BruneauPlutaSaxe2011_Stimuli_0.pdf` (16 pages). Parsed by `code/experiments/56_w7_bruneau.py --parse` into `dataset/bruneau/bruneau_stimuli.csv`.

| condition | domain | harm | n items | median words |
|---|---|---:|---:|---:|
| PP | physical | 1 | 24 | 46 |
| PPC | physical | 0 | 24 | 46 |
| EP | emotional | 1 | 24 | 47 |
| EPC | emotional | 0 | 24 | 47 |
| FBP | false_belief | 1 | 24 | 46 |
| FBC | false_belief | 0 | 24 | 45 |

Total 144 stimuli, 72 item pairs. Unpaired items: none.

## Matched-pair check (verbatim, one pair per domain)

The two members of a pair share their opening and diverge only at the ending. This is why folds and bootstrap resamples are over pairs.

### PP/PPC (physical), item 01

- **PP** (harm=1): Joe was playing soccer with his friends. He slid in to steal the ball away, but his cleat stuck in the grass and he rolled over his ankle, breaking his ankle and tearing the ligaments. His face was flushed as he rolled over.

- **PPC** (harm=0): Joe was playing soccer with his friends. He slid in to steal the ball away, and he kicked the ball away from the opposing player, got to his feat and began dribbling down the field. His face was flushed as he ran.

### EP/EPC (emotional), item 01

- **EP** (harm=1): Ron and his wife recently became a foster parents to a 10 year old girl. Ron promises her that he will be at a concert that she is playing in. On the way to the concert Ron is caught in traffic. When he arrives the concert is already over.

- **EPC** (harm=0): Ron and his wife recently became a foster parents to a 10 year old girl. Ron promises her that he will be at a concert that she is playing in. On the way to the concert Ron buys some flowers. When he arrives the concert is just about to begin.

### FBP/FBC (false belief), item 01

- **FBP** (harm=1): Tyler just became a step-father of a teenaged daughter. It seems that Tyler is finally being accepted by her. Walking by her room he heard her complaining about how much she hated him. Actually she was talking about a boy at school named Tyler.

- **FBC** (harm=0): Tyler just became a step-father of a teenaged daughter. It seems that Tyler is finally being accepted by her. Walking by her room he heard her talking to her friend on the phone. Actually she was practicing her lines for a drama performance.
