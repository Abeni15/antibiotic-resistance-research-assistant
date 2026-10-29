import unittest
from unittest.mock import patch
from agents import _extractive,_fallback_queries
class WorkflowTests(unittest.TestCase):
 def test_five_evaluation_questions_defined(self):
  from evaluation import QUESTIONS
  self.assertEqual(len(QUESTIONS),5)
 def test_extractive_claim_keeps_verbatim_source(self):
  papers=[{"pmid":"1","abstract":"A meaningful finding is described in this abstract. Another line."}]
  claim=_extractive(papers)[0]
  self.assertEqual(claim["supporting_passages"]["1"],"A meaningful finding is described in this abstract.")
 def test_question_specific_query(self):
  self.assertIn("antimicrobial susceptibility testing",_fallback_queries("What are limitations of diagnostic methods?")[0])
if __name__=="__main__":unittest.main()

