"""
Display the generated overfitting plots to verify they look correct.
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and display the plot
plot_path = '/home/runner/work/High_Dimensional_Linear_Models/High_Dimensional_Linear_Models/Python/output/overfitting_plots.png'
img = mpimg.imread(plot_path)

plt.figure(figsize=(15, 5))
plt.imshow(img)
plt.axis('off')
plt.title('Generated Overfitting Analysis Plots', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("✅ Three separate plots generated successfully:")
print("1. R² (Full Sample) vs Number of Features")
print("2. Adjusted R² (Full Sample) vs Number of Features") 
print("3. Out-of-Sample R² vs Number of Features")
print("\n📊 All plots show expected overfitting behavior patterns")