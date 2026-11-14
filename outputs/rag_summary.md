# Step 5a - Retrieval-selected excerpts (RAG comparison arm)

Firms processed: 34  |  Prompt version: v1.0-rag

Overlap = share of selected characters that Step 3's fixed head+middle excerpt would also have shown the LLM. Low overlap means retrieval is surfacing material the baseline arm never sees.

- Mean MD&A passages per firm: 6.9
- Mean Risk passages per firm: 4.3
- Mean MD&A overlap with Step 3 excerpt: 0.46
- Mean Risk overlap with Step 3 excerpt: 0.42

Ratings for this arm are produced by the Step 3 runner once an API key is provided (same schema, same verification). This step generates NO ratings on its own.

    cik                                name  label  mda_passages  risk_passages  mda_overlap  risk_overlap
 886158       20230930-DK-Butterfly-1, Inc.      1             4              5        1.000         0.000
  47129                          HERTZ CORP      1             5              5        1.000         0.000
1255474                Whiting Holdings LLC      1             8              5        0.250         0.767
1310067                 SEARS HOLDINGS CORP      1             8              2        0.250         1.000
1004980                           PG&E Corp      1             8              5        0.000         0.000
1282266           WINDSTREAM HOLDINGS, INC.      1             8              5        0.000         0.000
  84129                   NEW RITE AID, LLC      1             7              3        0.429         1.000
 719739                 SVB FINANCIAL GROUP      1             8              5        0.229         0.231
1813756                         WeWork Inc.      1             8              5        0.000         0.000
1567892            Keenova Therapeutics plc      1             8              4        0.854         1.000
1326428                      Pressburg, LLC      1             4              1        1.000         1.000
1349436                SANDRIDGE ENERGY INC      1             8              3        0.625         1.000
 887921                     REVLON INC /DE/      1             3              5        1.000         0.172
 216085     HAVERTY FURNITURE COMPANIES INC      0             8              5        0.125         0.506
 764478                     BEST BUY CO INC      0             8              5        0.698         0.944
1821159                           EVgo Inc.      0             8              5        0.000         0.200
1070412                  CNX Resources Corp      0             8              5        0.000         0.200
1631463                  Barrel Energy Inc.      0             3              5        1.000         0.200
  28917                     DILLARD'S, INC.      0             3              5        1.000         0.000
  18654                  Ameren Illinois Co      0             8              5        0.000         0.467
 811156                     CMS ENERGY CORP      0             8              5        0.185         0.000
1051512   TELEPHONE & DATA SYSTEMS INC /DE/      0             8              5        0.021         0.000
1304492                        Anterix Inc.      0             8              5        0.250         0.200
1040130                  PETMED EXPRESS INC      0             8              5        0.125         0.338
 715579                           ACNB CORP      0             8              5        0.603         0.190
1462120           Live Oak Bancshares, Inc.      0             7              5        1.000         0.584
1666291         Claros Mortgage Trust, Inc.      0             8              5        0.723         1.117
1680132               CANNABIS SUISSE CORP.      0             3              3        1.000         1.000
1430306 Tonix Pharmaceuticals Holding Corp.      0             8              5        0.104         0.000
1498382         TuHURA Biosciences, Inc./NV      0             8              5        0.446         0.167
1070412                  CNX Resources Corp      0             8              1        0.375         1.000
1631463                  Barrel Energy Inc.      0             3              0        1.000         0.000
1652561                          DSwiss Inc      0             8              3        0.375         1.000
1840199                      Waldencast plc      0             8              5        0.072         0.077