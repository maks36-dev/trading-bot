from tradeBot import TradeBot
from model import Model
import time

def main(currency: str, time_sleep: int):
    bot = TradeBot()
    bot.set_symbol(currency+"USDT")

    model = Model(bot.get_data(500, uni=False), future=1, past_history=30)
    
    quantity = 0.01

    sell_flag = False
    buy_flag = False
    steps = 10 # количество действий на бирже

    for step in range(1, steps):
        print(f"Step: {step}")
        print("USDT balance:", bot.get_balance()["USDT"])

        data = bot.get_data(30, uni=False)
        future_price = model.predict(data)

        current_price = bot.current_price()

        if sell_flag:
            if current_price > future_price:
                print("Continue sell")
            else:
                sell_flag = False
                bot.new_order(quantity, "BUY")
                bot.new_order(quantity, "BUY")
                buy_flag = True
        elif buy_flag:
            if current_price < future_price:
                print("Continue buy")
            else:
                buy_flag = False
                bot.new_order(quantity, "SELL")
                bot.new_order(quantity, "SELL")
                sell_flag = True
        else:
            if future_price > current_price:
                bot.new_order(quantity, "BUY")
                buy_flag = True
            else:
                bot.new_order(quantity, "SELL")
                sell_flag = True

        print()
        time.sleep(time_sleep)




if __name__ == "__main__":
    currency = "BTC"
    time_sleep = 10 # задержка

    main(currency, time_sleep)



