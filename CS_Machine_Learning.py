from sklearn import tree

clf = tree.DecisionTreeClassifier()

# Assumption keyword in CS log
# 1 = i am Unhappy
# 2 = feel frustrated
# 3 = I need to talk to your supervisor
# 4 = I filled a complaint

# [Keyword1, Keyword2, Keyword3] ** this is data set, we dont need to idenify by human **
X = [[1, 2, 1], [3, 1, 2], [2, 1, 1], [1, 1, 1], [3, 3, 1],
     [2, 3, 3], [3, 3, 2],[1, 1, 1], [2, 2, 2], [4, 3, 3], [4, 4, 4]]

# This is result of above data
Y = ['bad', 'bad', 'bad', 'bad', 'bad', 'very bad', 'very bad', 'very bad',
     'bad', 'bad', 'Highly alert and report to supervisor']


# Teach the machine using X keyword and Y result to predict result
clf = clf.fit(X, Y)

prediction = clf.predict([[9, 9, 9]])

# Preduct outcome

print(prediction)
