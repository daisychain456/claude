# Beginner's Testing Guide - No Coding Experience Required!

Welcome! This guide will walk you through testing the language retention predictor, even if you've never written code before.

## What You'll Do

You'll run 4 simple commands that will:
1. Install the required software libraries
2. Create sample data about 500 children
3. Train the AI to make predictions
4. Create pretty graphs to show the results

**Time needed:** About 5 minutes

---

## Step 1: Open Your Terminal

**What is a terminal?** It's a window where you type commands to talk to your computer.

### How to open it:

**On Mac:**
- Press `Command + Space`
- Type "Terminal"
- Press Enter

**On Windows:**
- Press `Windows key`
- Type "Command Prompt" or "PowerShell"
- Press Enter

**On Linux:**
- Press `Ctrl + Alt + T`

You should see a window with a blinking cursor. This is your terminal!

---

## Step 2: Go to the Project Folder

**What does this do?** Tells your computer where to find the project files.

**Type this command and press Enter:**

```bash
cd /home/user/claude
```

**What you'll see:** The cursor moves to the next line. That's it!

---

## Step 3: Install Required Libraries

**What does this do?** Installs the free software tools needed to run the predictor.

**Type this command and press Enter:**

```bash
pip install -r requirements.txt
```

**What you'll see:**
- Text scrolling by as it downloads and installs things
- Some yellow warnings (these are normal - ignore them!)
- Takes about 30-60 seconds

**Wait for it to finish** - you'll see your cursor come back.

---

## Step 4: Generate Sample Data

**What does this do?** Creates pretend data about 500 children to test with. This includes things like their age, family situation, and whether they speak their parent's language.

**Type this command and press Enter:**

```bash
python generate_sample_data.py
```

**What you'll see:**

```
Generating synthetic language retention data...

✓ Generated 500 samples
✓ Saved to: data/language_retention_data.csv

============================================================
DATA SUMMARY
============================================================

Target variable distribution:
  Speaks parent language: 461 (92.2%)
  Does not speak parent language: 39 (7.8%)

Age range: 5 - 18 years
Average parents' years in country: 14.8 years
...
```

**This means:** Success! It created 500 fake children's data to practice with.

---

## Step 5: Train the AI Model

**What does this do?** Teaches the AI to recognize patterns in the data so it can make predictions.

**Type this command and press Enter:**

```bash
python train_model.py
```

**What you'll see:**

```
============================================================
LANGUAGE RETENTION PREDICTION MODEL TRAINING
============================================================
Loading data from data/language_retention_data.csv...
✓ Loaded 500 samples with 22 columns
...

Training Logistic Regression...
  Test Accuracy: 0.7700
  Test ROC-AUC: 0.8383

Training Random Forest...
  Test Accuracy: 0.9300
  Test ROC-AUC: 0.8546

✓ Best model: Random Forest
  ROC-AUC: 0.8546
...
```

**This means:**
- The AI tested 3 different prediction methods
- "Random Forest" worked best with 93% accuracy
- The model has been saved and is ready to use!

**Takes about 10-15 seconds**

---

## Step 6: Create Pretty Graphs

**What does this do?** Makes visual charts showing how well the predictor works and which factors matter most.

**Type this command and press Enter:**

```bash
python visualize_results.py
```

**What you'll see:**

```
============================================================
GENERATING VISUALIZATIONS
============================================================

✓ Saved feature importance plot: results/feature_importance.png
✓ Saved confusion matrix: results/confusion_matrix.png
✓ Saved ROC curve: results/roc_curve.png
✓ Saved demographic patterns: results/demographic_patterns.png
✓ Saved correlation heatmap: results/correlation_heatmap.png
✓ Saved retention by categories: results/retention_by_categories.png

VISUALIZATION COMPLETE!
```

**This means:** Success! 6 graphs were created showing the results.

---

## Step 7: Test Predictions on New Children

**What does this do?** Uses the trained AI to predict whether 3 new children will speak their parent's language.

**Type this command and press Enter:**

```bash
python predict.py --input test_new_data.csv --output test_predictions.csv
```

**What you'll see:**

```
============================================================
LANGUAGE RETENTION PREDICTION
============================================================

Predicted to speak parent language: 2 (66.7%)
Predicted NOT to speak parent language: 1 (33.3%)
Average confidence: 75.8%

SAMPLE PREDICTIONS (first 5 rows)
child_id  predicted_speaks_parent_language  probability_yes  confidence
     501                                 1         0.920270    92.0%
     502                                 0         0.444890    55.5%
     503                                 1         0.799868    80.0%
```

**This means:**
- Child 501: **Will speak** parent language (92% confident)
- Child 502: **Won't speak** parent language (55% confident)
- Child 503: **Will speak** parent language (80% confident)

---

## 🎉 Congratulations! You Did It!

You just:
1. ✓ Installed the software
2. ✓ Generated sample data (500 children)
3. ✓ Trained an AI model (93% accuracy!)
4. ✓ Created visualizations
5. ✓ Made predictions on new data

---

## What to Look At Next

### 1. View the Generated Graphs

The graphs are saved in the `results/` folder. Here's how to view them:

**On Mac:**
```bash
open results/
```

**On Windows:**
```bash
explorer results
```

**On Linux:**
```bash
xdg-open results/
```

This opens a folder with 6 images. Double-click any image to see it!

### 2. Look at the Prediction Results

Want to see the full predictions? Open the file:

**Type:**
```bash
cat test_predictions.csv
```

This shows all the details about the 3 children and their predictions.

### 3. Look at the Sample Data

Want to see what the training data looks like?

**Type:**
```bash
head -n 5 data/language_retention_data.csv
```

This shows the first 5 children from the 500 sample dataset.

---

## Understanding the Results

### What the Graphs Show

1. **feature_importance.png** - Which factors matter most?
   - Top factor: Parents speaking heritage language at home
   - Shows the 15 most important factors

2. **confusion_matrix.png** - How accurate is the predictor?
   - Shows correct vs incorrect predictions

3. **roc_curve.png** - Overall performance score
   - Higher curve = better predictor
   - Score of 0.855 is very good!

4. **demographic_patterns.png** - How do factors differ?
   - Compares children who speak vs don't speak

5. **correlation_heatmap.png** - Which factors relate to each other?
   - Red = factors go together
   - Blue = factors go opposite ways

6. **retention_by_categories.png** - Success rates by situation
   - Shows % who speak based on different factors

---

## Common Questions

### "I see yellow WARNING messages - is that bad?"

**No!** Yellow warnings are normal. Only red ERROR messages are problems.

### "The numbers are different from the guide"

**That's fine!** The sample data is random, so you'll get slightly different numbers each time.

### "How do I close the terminal?"

Just type `exit` and press Enter, or click the X button.

### "Can I run these commands again?"

Yes! You can repeat any step. If you run `train_model.py` again, it will retrain with the same data.

### "What if I get an error?"

1. Make sure you're in the right folder (Step 2)
2. Make sure you installed the libraries (Step 3)
3. Check that you typed the command exactly as shown
4. Copy the error message and ask for help!

---

## What's Actually Happening? (Optional - For the Curious)

**Step 3 (Install libraries):**
- Downloads free Python tools like pandas (data handling) and scikit-learn (AI)

**Step 4 (Generate data):**
- Creates 500 rows of realistic fake data
- Each row = one child with 20 pieces of information
- Uses probability to make it realistic (e.g., families near ethnic neighborhoods are more likely to speak heritage language)

**Step 5 (Train model):**
- Teaches 3 different AI algorithms to spot patterns
- Tests each one to see which is most accurate
- Saves the best one (Random Forest in this case)

**Step 6 (Visualizations):**
- Reads the results from Step 5
- Creates graphs using matplotlib (graphing library)
- Saves as PNG image files

**Step 7 (Predictions):**
- Loads the trained AI from Step 5
- Reads new children's data from CSV file
- Calculates probability for each child
- Saves results to a new CSV file

---

## Next Steps

Now that you've tested it, you could:

1. **Look at the documentation:**
   - `README.md` - Full explanation of all features
   - `USAGE_GUIDE.md` - More detailed examples

2. **Try with your own data:**
   - Look at `data_template.csv` to see the format
   - Fill in real data about children you're studying
   - Run `train_model.py` with your data

3. **Modify the visualizations:**
   - Colors, sizes, labels can all be changed
   - Edit `visualize_results.py` (or ask for help!)

---

## Quick Reference - All Commands

If you want to run everything again from scratch:

```bash
cd /home/user/claude
pip install -r requirements.txt
python generate_sample_data.py
python train_model.py
python visualize_results.py
python predict.py --input test_new_data.csv --output test_predictions.csv
```

Copy and paste each line one at a time!

---

## Need Help?

If you get stuck:
1. Check which step you're on
2. Make sure previous steps completed successfully (look for ✓ checkmarks)
3. Copy any error messages
4. The files `README.md` and `USAGE_GUIDE.md` have more details

**Remember:** Everyone starts as a beginner! You're doing great! 🌟
