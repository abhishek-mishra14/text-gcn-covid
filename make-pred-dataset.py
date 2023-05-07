import pandas as pd

def concatenate(title, keyword, abstract):
    title = f"{title}"
    if type(keyword) == float:
        # print("Keyword: ", keyword)
        keyword = ""
    else:
        keyword = keyword.replace(";", " ")
        keyword = f"{keyword}."

    abstract = f"{abstract}"

    return title + "$" + keyword + "$" + abstract

df = pd.read_excel("dataset.xlsx")
df.dropna(subset=["Article title", "Article abstract"], inplace=True)
df.fillna({"Contextual": -1}, inplace=True)


df["Concatenated"] = df.apply(
    lambda x: concatenate(
        x["Article title"], x["Article keywords"], x["Article abstract"]
    ),
    axis=1,
)
df = df.astype({"Contextual": "int32"})
df = df[["Concatenated", "Contextual"]]

df_labeled = df[df["Contextual"] != -1]
df_unlabeled = df[df["Contextual"] == -1]

df_labeled.reset_index(drop=True, inplace=True)
df_unlabeled.reset_index(drop=True, inplace=True)

print("Labeled: ", len(df_labeled))
print("Unlabeled: ", len(df_unlabeled))

print("Labeled: ", len(df_labeled))
print("Unlabeled: ", len(df_unlabeled))

pred_labels = []

with open("data/corpus/covid-semi_pred.txt", "r") as f:
    for line in f:
        label = line.strip().split("\t")[3]
        pred_labels.append(label)
print(len(pred_labels))
print(len(df_unlabeled))


for idx, row in df_unlabeled.iterrows():
    try:
        row["Contextual"] = pred_labels[idx]  
    except IndexError as e:
        print(idx, e)
        break  

df = pd.concat([df_labeled, df_unlabeled])

df.to_csv("data/corpus/covid-semi_pred.csv", index=False)