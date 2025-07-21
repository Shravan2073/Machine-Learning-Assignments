import pandas as pd

def a3():
    df = pd.read_excel('LB_2.xlsx', sheet_name='IRCTC Stock Price')
    price = pd.to_numeric(df["Price"], errors="coerce")
    chg = pd.to_numeric(df["Chg%"], errors="coerce")
    mean_price = price.mean()
    var_price = price.var()
    mean_wed = price[df["Day"]=="Wed"].mean()
    mean_apr = price[df["Month"]=="Apr"].mean()
    loss_prob = (chg < 0).mean()
    profit_wed_prob = ((chg > 0) & (df["Day"]=="Wed")).sum() / (df["Day"] == "Wed").sum()
    cond_profit_prob = profit_wed_prob / ((df["Day"] == "Wed").sum() / len(df))
    # Scatter for plt (optional)
    scatter_x = df["Day"]
    scatter_y = chg
    return mean_price, var_price, mean_wed, mean_apr, loss_prob, profit_wed_prob, cond_profit_prob, (scatter_x, scatter_y)

if __name__ == '__main__':
    output_a3 = a3()
    print("A3 Output:", output_a3)
