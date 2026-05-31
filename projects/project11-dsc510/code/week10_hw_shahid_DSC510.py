# DSC 510
# Week 10
# Programming Assignment Week 10
# Author: Komal Shahid
# 08/12/2023
# *******************************Cash Register ********************************************************
import locale
class CashRegister:
    """
    This is the class cash register which contains the item counts and total price of the items.
    """

    def __init__(self):
        self.totalPrice = 0
        self.itemCount = 0

    def addItem(self, price):
        self.totalPrice += price
        self.itemCount += 1

    def getTotal(self):
        return self.totalPrice

    def getCount(self):
        return self.itemCount

    def clearcart(self):
        self.totalPrice = 0.0
        self.itemCount = 0


def main():
    print("Hello, Welcome to your Cash Register. Spend Wisely!")
    register = CashRegister()
    locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
    while True:
        user_input = input("Please enter a price for your item or done when you are finished:")
        if user_input == 'done': break
        try:
            price = float(user_input)
            register.addItem(price)
        except ValueError as e:
            print("Price is not valid. please try again.")

    print ("Cart Overview:")
    print ("**********************************************")
    print (f"The total number of in the cart are {register.getCount ()}.\n"
           f"The total Price for you items is {locale.currency (register.getTotal ())}.")
    register.clearcart ()
    print("Resetting the cart" )


if __name__ == '__main__':
    main ()

#**********************************************END********************************************************************