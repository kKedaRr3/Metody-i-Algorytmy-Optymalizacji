set of int: ITEM;  
int: capacity1;  % Pojemność plecaka 1
int: capacity2;  % Pojemność plecaka 2
int: capacity3;  % Pojemnosc plecaka 3

array[ITEM] of int: profits;
array[ITEM] of int: weights;

var set of ITEM: knapsack1;  % Przedmioty w placaku 1
var set of ITEM: knapsack2;  % Przedmioty w placaku 2
var set of ITEM: knapsack3;  % Przedmioty w placaku 3

% Ograniczenie na wagę przedmiotów w plecaku
constraint sum(i in knapsack1)(weights[i]) <= capacity1;
constraint sum(i in knapsack2)(weights[i]) <= capacity2;
constraint sum(i in knapsack3)(weights[i]) <= capacity3;

% Ograniczenie: każdy przedmiot może być tylko w jednym plecaku
constraint forall(i in ITEM) (
    (i in knapsack1) + (i in knapsack2) + (i in knapsack3) <= 1
);

% Funkcja celu: maksymalizacja sumy wartości przedmiotów w plecaku
% solve maximize sum(i in knapsack)(profits[i]);
solve maximize sum(i in knapsack1)(profits[i]) + sum(i in knapsack2)(profits[i]) + sum(i in knapsack3)(profits[i]);

output [
    "knapsack1 = \(knapsack1)\n",
    "knapsack2 = \(knapsack2)\n",
    "knapsack3 = \(knapsack3)\n",
    "total profit = \(sum(i in knapsack1)(profits[i]) + sum(i in knapsack2)(profits[i]) + sum(i in knapsack3)(profits[i]))\n"
];
