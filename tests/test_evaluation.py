import unittest
from evaluation import QUESTIONS
class EvaluationTests(unittest.TestCase):
 def test_minimum_questions(self):
  self.assertEqual(len(QUESTIONS),5)
  self.assertEqual(len(set(QUESTIONS)),5)
  self.assertTrue(all("escherichia coli" in question.lower() or "e. coli" in question.lower() for question in QUESTIONS))
  self.assertIn("antibiotic resistance",QUESTIONS[0].lower())
  self.assertIn("small number",QUESTIONS[-1])
if __name__=="__main__":unittest.main()
