# Data dictionary

`cancer_reg.csv` has one row per US county (3,047 rows). The data was aggregated from the American Community Survey (census.gov), clinicaltrials.gov, and cancer.gov. Column names below match the CSV header.

| Column | Description | Source |
|---|---|---|
| `TARGET_deathRate` | Target, renamed to `mortality_rate` on load. Mean cancer mortalities per 100,000 people | (a) |
| `avgAnnCount` | Mean number of reported cancer cases diagnosed annually | (a) |
| `avgDeathsPerYear` | Mean number of reported cancer mortalities | (a) |
| `incidenceRate` | Mean cancer diagnoses per 100,000 people | (a) |
| `medIncome` | Median income per county (called `medianIncome` in the original dictionary) | (b) |
| `popEst2015` | Population of the county | (b) |
| `povertyPercent` | Percent of the population in poverty | (b) |
| `studyPerCap` | Cancer-related clinical trials per capita | (a) |
| `binnedInc` | Median income per capita binned by decile | (b) |
| `MedianAge` | Median age of county residents | (b) |
| `MedianAgeMale` | Median age of male residents | (b) |
| `MedianAgeFemale` | Median age of female residents | (b) |
| `Geography` | County name | (b) |
| `AvgHouseholdSize` | Mean household size | (b) |
| `PercentMarried` | Percent of residents who are married | (b) |
| `PctNoHS18_24` | Percent of residents aged 18 to 24 whose highest education is less than high school | (b) |
| `PctHS18_24` | Percent aged 18 to 24 with a high school diploma as highest education | (b) |
| `PctSomeCol18_24` | Percent aged 18 to 24 with some college as highest education | (b) |
| `PctBachDeg18_24` | Percent aged 18 to 24 with a bachelor's degree as highest education | (b) |
| `PctHS25_Over` | Percent aged 25 and over with a high school diploma as highest education | (b) |
| `PctBachDeg25_Over` | Percent aged 25 and over with a bachelor's degree as highest education | (b) |
| `PctEmployed16_Over` | Percent aged 16 and over who are employed | (b) |
| `PctUnemployed16_Over` | Percent aged 16 and over who are unemployed | (b) |
| `PctPrivateCoverage` | Percent with private health coverage | (b) |
| `PctPrivateCoverageAlone` | Percent with private health coverage alone (no public assistance) | (b) |
| `PctEmpPrivCoverage` | Percent with employer-provided private health coverage | (b) |
| `PctPublicCoverage` | Percent with government-provided health coverage | (b) |
| `PctPublicCoverageAlone` | Percent with government-provided health coverage alone | (b) |
| `PctWhite` | Percent who identify as White | (b) |
| `PctBlack` | Percent who identify as Black | (b) |
| `PctAsian` | Percent who identify as Asian | (b) |
| `PctOtherRace` | Percent who identify as a race other than White, Black, or Asian | (b) |
| `PctMarriedHouseholds` | Percent of married households | (b) |
| `BirthRate` | Live births relative to the number of women in the county | (b) |

Sources: (a) years 2010 to 2016; (b) 2013 Census estimates.
