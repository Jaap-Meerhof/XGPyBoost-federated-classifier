import matplotlib.pyplot as plt

# Create a figure and axes
fig, ax = plt.subplots()

# Add text using the text() function
ax.text(0.5, 0.5, "This is a lot of text!\nLine 1\nLine 2\nLine 3\n...", 
        horizontalalignment='center', verticalalignment='center',
        fontsize=12, color='black')

# Set the axis limits (optional)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

# Remove axis ticks (optional)
ax.set_xticks([])
ax.set_yticks([])

# Show the plot
plt.show()