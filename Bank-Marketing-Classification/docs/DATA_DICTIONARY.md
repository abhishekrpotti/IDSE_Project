# Data dictionary — UCI Bank Marketing (`bank-full.csv`)

Target: `y` (yes/no term-deposit subscription). Primary scenario: predict before a marketing call. Exclude `duration` for this scenario.

| Column | Meaning | Type / notes | Timing note |
|---|---|---|---|
| age | Client age | Integer | Candidate pre-call |
| job | Job category | Categorical; retain `unknown` as a documented value | Candidate pre-call |
| marital | Marital status; divorced includes divorced or widowed per source | Categorical | Candidate pre-call |
| education | Education level | Categorical; retain `unknown` | Candidate pre-call |
| default | Credit in default? | yes/no | Candidate pre-call |
| balance | Average yearly balance in euros | Integer; negative and large positive values occur | Candidate pre-call |
| housing | Housing loan? | yes/no | Candidate pre-call |
| loan | Personal loan? | yes/no | Candidate pre-call |
| contact | Communication type | Categorical; retain `unknown` | Confirm channel is known at list scoring |
| day | Day of month of last contact | Integer, 1–31 | Confirm availability when scoring |
| month | Month of last contact | Categorical | Confirm availability when scoring |
| duration | Duration of current call in seconds | Integer | Post-call only; leakage for pre-call prediction; exclude |
| campaign | Contacts performed during current campaign, including last contact | Integer | Potentially unavailable before call; confirm |
| pdays | Days since last contact in a previous campaign | Integer; **-1 means not previously contacted**, not negative elapsed time | Prior campaign history |
| previous | Contacts before current campaign | Integer; 0 means none | Prior campaign history |
| poutcome | Previous campaign outcome | Categorical; `unknown` retained | Prior campaign history |
| y | Subscribed a term deposit? | Binary target, yes/no | Outcome |

The downloaded UCI archive includes `bank-names.txt`, which documents these fields; acquisition details are in `data/README.md`. UCI Bank Marketing, DOI 10.24432/C5K306. The source reports no missing attribute values; strings such as `unknown` are categorical values, not blank cells.
