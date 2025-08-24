---
title: "Introduction to Linear Algebra"
datePublished: Sun Aug 24 2025 04:52:24 GMT+0000 (Coordinated Universal Time)
cuid: cmep7p2l5000302jp0ltybhih
slug: introduction-to-linear-algebra
tags: linear-algebra

---

## 1\. Basic Mathematical Concepts

Before diving into linear algebra, let’s first review some essential concepts: **sets, mappings, and functions.**

---

## Set

A **set** is a collection of objects that satisfy certain conditions.  
Each object belonging to a set is called an **element**.

**Examples:**

* The set of natural numbers 1, 2, 3 → {1, 2, 3}
    
* The set of odd numbers → {1, 3, 5, 7, …}
    

### Ways to Represent a Set

* **Roster Form**  
    List out each element of the set.
    
    * Example: A = {1, 2, 3, 4}
        
* **Set-builder Form**  
    Specify the property that the elements must satisfy.
    
    * Example: B = { x | x is a natural number less than or equal to 10 }
        

---

## Basic Concepts

* **Subset**  
    If every element of set A is also in set B, then A is a subset of B.
    
    * Notation: A ⊆ B
        
    * Example: {1, 2} ⊆ {1, 2, 3}
        
* **Empty Set**  
    A set with no elements.
    
    * Notation: ∅ or {}
        
    * Example: {x | x is an integer greater than 5 and less than 4} = ∅
        

---

## Set Operations

* **Union**  
    The set containing all elements from both sets.
    
    * Notation: A ∪ B
        
    * Example: {1, 2} ∪ {2, 3} = {1, 2, 3}
        
* **Intersection**  
    The set of elements common to both sets.
    
    * Notation: A ∩ B
        
    * Example: {1, 2} ∩ {2, 3} = {2}
        
* **Difference**  
    The set of elements in one set but not in the other.
    
    * Notation: A – B
        
    * Example: {1, 2, 3} – {2, 3} = {1}
        

---

## Function / Mapping

A **mapping (or function)** assigns each element of a set A to exactly one element of a set B.

* A: **Domain**
    
* B: **Codomain**
    
* The actual set of mapped values in B: **Range (Image)**
    

**Example:**  
f: A → B, f(x) = x²

* A = {1, 2, 3} (domain)
    
* B = {1, 4, 9, 16, 25} (codomain)
    
* Range = {1, 4, 9}
    

---

## Types of Functions

* **Surjective (Onto)**  
    The range is equal to the codomain.  
    → Every element in the codomain is mapped.
    
* **Injective (One-to-one)**  
    Different inputs always map to different outputs.  
    → If x₁ ≠ x₂, then f(x₁) ≠ f(x₂).
    
* **Bijective (One-to-one Correspondence)**  
    Both injective and surjective.  
    → A perfect one-to-one mapping between the domain and codomain.
    

---

## Inverse Function

If a function f: A → B is bijective, then there exists an **inverse mapping** g: B → A.  
This is called the **inverse function** of f and is denoted by f⁻¹.

**Example:**

* f(x) = 2x (domain = all real numbers, codomain = all real numbers)
    
* Inverse: f⁻¹(x) = x/2
    

## Matrix

A **matrix** is an arrangement of numbers or symbols in a rectangular form.  
Matrices are mainly used in linear algebra to represent data or linear transformations.

**Example:**

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1756010507890/8fd29762-e445-4c9e-9563-8b46dda33bc1.png align="center")

---

### Basic Terminology

* **Element / Entry**  
    Each individual number contained in a matrix.
    
    * Example: In matrix A, a12=2a\_{12} = 2a12​=2 (the element in the 1st row and 2nd column).
        
* **Square Matrix**  
    A matrix in which the number of rows equals the number of columns.
    
    * Example: a 3×3 matrix.
        
* **Main Diagonal**  
    The diagonal running from the top-left to the bottom-right of a square matrix.
    
    * Example: a11,a22,a33a\_{11}, a\_{22}, a\_{33}a11​,a22​,a33​.
        
* **Diagonal Matrix**  
    A square matrix in which all the elements outside the main diagonal are zero.
    
    ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1756010399781/2e15b309-9c9a-4730-a55d-3004fb9779bf.png align="center")
    
* **Identity Matrix (Unit Matrix)**  
    A square matrix whose main diagonal elements are all 1 and whose other elements are all 0.
    
* ![](https://cdn.hashnode.com/res/hashnode/image/upload/v1756010475427/81a216e7-83d6-46f0-b992-c876ad40bf0e.png align="center")
    

## Vector

A **vector** refers to a matrix that has only a single row or a single column.

* A vector consisting of a single row is called a **row vector**.
    
* A vector consisting of a single column is called a **column vector**.
    

# 2\. Applications of Linear Algebra

Linear algebra is not just about learning how to manipulate matrices. In reality, it serves as the **fundamental language of computer science, data science, and engineering**. Let’s take a look at some representative applications.

* **Data Representation and Processing**  
    Vectors and matrices provide the basic framework for storing and operating on structured data. For instance, in machine learning, input datasets are typically represented as matrices.
    
* **Webpage Ranking**  
    Google’s PageRank algorithm models the web’s link structure as a matrix and evaluates the importance of each page through eigenvector computations.
    
* **Computer Graphics**  
    3D modeling and transformations such as rotation, translation, and scaling are all expressed as matrix multiplications. Modern graphics in games and movies rely heavily on linear algebra.
    
* **Robotics**  
    The position and orientation of robotic arms or drones are handled through coordinate transformations, which are calculated using matrix operations.
    
* **Electrical Circuit Analysis**  
    The equations governing a circuit’s nodes form systems of linear equations, which can be expressed and solved using matrices.
    
* **Fourier Transform**  
    Decomposing signals into frequency components via the Fourier transform can be viewed as a linear operation, and it is fundamental in image and audio processing.
    
* **Dimensionality Reduction (PCA)**  
    Principal Component Analysis (PCA) reduces unnecessary dimensions in data analysis, relying on eigenvalue decomposition.
    
* **Multivariate Gaussian Distribution**  
    The covariance matrix captures correlations between variables, enabling the modeling of multidimensional probability distributions—widely used in machine learning and statistics.
    
* **Kalman Filter**  
    An algorithm that filters noise and estimates hidden states from sensor data. It is based on matrix representations of linear dynamic systems.