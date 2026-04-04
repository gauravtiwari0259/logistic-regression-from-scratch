class Rectangle:
    def __init__(self, width, height, color):
        self.width = width
        self.height = height
        self.color = color
        print(f"Rectangle created: {self.color}, {self.width}x{self.height}")

    def drawRectangle(self):
        # This method would contain the actual code to render the graphic,
        # using a graphics library (e.g., plot commands).
        print("-" * 30)
        print("Drawing graphic...")
        print(f"Drawing a {self.color} rectangle with dimensions {self.width}x{self.height}.")
        # Placeholder visual representation:
        for _ in range(self.height):
            print(f"[{'=' * self.width}]")
        print("-" * 30)

# 1. Create the Object (Instance of the Class)
FatYellowRectangle = Rectangle(20, 5, 'yellow')

# 2. Accessing Attributes (Data)
print(FatYellowRectangle.height)

print(FatYellowRectangle.width)

print(FatYellowRectangle.color)

# 3. Calling the Method (Behavior)
FatYellowRectangle.drawRectangle()
