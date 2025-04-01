import marimo

__generated_with = "0.11.17"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    df = pd.read_csv("data_nfcorpus/data.csv")

    return (df,)


@app.cell
def _(df):
    df[df['query_id']=='PLAIN-100']
    return


@app.cell
def _(df):
    df_sampled = df.groupby(['query_id', 'relevance']).apply(lambda x: x.sample(1)).reset_index(drop=True)
    return (df_sampled,)


@app.cell
def _(df_sampled):

    df1 = df_sampled.groupby("query_id")
    df1.head()
    return (df1,)


if __name__ == "__main__":
    app.run()
