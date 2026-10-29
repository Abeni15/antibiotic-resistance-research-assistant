import unittest
from tools import validate_evidence
class EvidenceTests(unittest.TestCase):
 def setUp(self):self.sources=[{"pmid":"22","title":"Paper","abstract":"The efflux pump AcrB contributes to resistance."}]
 def test_exact_and_normalized_passage(self):
  c={"source_ids":["22"],"supporting_passages":{"22":"THE efflux pump AcrB contributes to resistance"}}
  self.assertEqual(validate_evidence([c],self.sources)["results"][0]["status"],"verified_source_and_passage")
 def test_missing_source_and_passage(self):
  result=validate_evidence([{"text":"x"},{"source_ids":["99"],"supporting_passages":{"99":"x"}}],self.sources)
  self.assertEqual(result["results"][0]["status"],"missing_source");self.assertEqual(result["results"][1]["status"],"missing_source")
 def test_absent_passage(self):
  r=validate_evidence([{"source_ids":["22"],"supporting_passages":{"22":"unsupported sentence"}}],self.sources)
  self.assertEqual(r["results"][0]["status"],"passage_not_found")
 def test_no_entailment_claim(self):
  r=validate_evidence([{"source_ids":["22"],"supporting_passages":{"22":"The efflux pump AcrB contributes to resistance."}}],self.sources)
  self.assertIn("entailment",r["results"][0]["reason"])
if __name__=="__main__":unittest.main()
