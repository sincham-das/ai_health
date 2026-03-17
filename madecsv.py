import pandas as pd

rxnorm=pd.read_csv("RXNCONSO.RRF",
                   sep="|",
                   header=None,
                   index_col=0,
                   dtype=str,
)

columns = [
"RXCUI","LAT","TS","LUI","STT","SUI","ISPREF",
"RXAUI","SAUI","SCUI","SDUI","SAB","TTY",
"CODE","STR","SRL","SUPPRESS","CVF"
]

rxnorm.columns = columns
drug_names=rxnorm["STR"]

drugs=rxnorm[
    (rxnorm["LAT"] == "ENG") &
    (rxnorm["SAB"] == "RXNORM") &
    (rxnorm["TTY"].isin(["IN","PIN","BN"]))
]

medicine_list=drugs["STR"].tolist()
print(medicine_list[:20])

df = pd.DataFrame(medicine_list, columns=["medicine_name"])

df.to_csv("medicine_dictionary.csv", index=False)