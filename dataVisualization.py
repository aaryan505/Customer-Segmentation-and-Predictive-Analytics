# Visualize customer segments
import matplotlib.pyplot as plt
import seaborn as sns

# Scatter plot of customer segments
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_data, x='income', y='age', hue='cluster', palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Income')
plt.ylabel('Age')
plt.show()

# Confusion matrix for model evaluation
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
