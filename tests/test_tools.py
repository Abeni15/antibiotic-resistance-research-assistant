import json,unittest,urllib.error
from unittest.mock import patch
import tools
XML=b"""<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>12345</PMID><Article><ArticleTitle>Resistance in E. coli</ArticleTitle><Abstract><AbstractText>Exact supporting abstract passage.</AbstractText></Abstract><Journal><Title>Example Journal</Title></Journal><AuthorList><Author><ForeName>Ada</ForeName><LastName>Lee</LastName></Author></AuthorList><Journal><JournalIssue><PubDate><Year>2024</Year></PubDate></JournalIssue></Journal></Article></MedlineCitation><PubmedData><ArticleIdList><ArticleId IdType="doi">10.1000/example</ArticleId></ArticleIdList></PubmedData></PubmedArticle></PubmedArticleSet>"""
class ToolTests(unittest.TestCase):
 def test_search_pubmed_calls_esearch_then_efetch(self):
  with patch.object(tools,"_get",side_effect=[b'{"esearchresult":{"count":"1","idlist":["12345"]}}',XML]) as get:
   result=tools.search_pubmed("E. coli resistance",2)
  self.assertTrue(result["ok"]);self.assertEqual(result["papers"][0]["pmid"],"12345");self.assertIn("Exact supporting",result["papers"][0]["abstract"]);self.assertEqual(get.call_count,2)
 def test_search_validation(self):
  self.assertFalse(tools.search_pubmed("",5)["ok"]);self.assertFalse(tools.search_pubmed("x",51)["ok"])
 def test_empty_search_returns_empty_without_fetch(self):
  with patch.object(tools,"_get",return_value=b'{"esearchresult":{"count":"0","idlist":[]}}') as get:
   result=tools.search_pubmed("rare test query")
  self.assertTrue(result["ok"]);self.assertEqual(result["papers"],[]);self.assertEqual(get.call_count,1)
 def test_malformed_search_json_is_structured_failure(self):
  with patch.object(tools,"_get",return_value=b"not json"):
   self.assertFalse(tools.search_pubmed("x")["ok"])
 def test_fetch_details(self):
  with patch.object(tools,"_get",return_value=XML):
   result=tools.fetch_paper_details("12345")
  self.assertTrue(result["ok"]);self.assertEqual(result["paper"]["doi"],"10.1000/example")
 def test_bad_and_missing_identifier(self):
  self.assertFalse(tools.fetch_paper_details("abc")["ok"])
  with patch.object(tools,"_get",return_value=b"<PubmedArticleSet/>"):
   self.assertFalse(tools.fetch_paper_details("12345")["ok"])
 def test_malformed_xml_is_structured_failure(self):
  with patch.object(tools,"_get",return_value=b"<not-closed"):
   self.assertFalse(tools.fetch_paper_details("12345")["ok"])
 def test_network_errors_retry_three_times(self):
  with patch.object(tools.urllib.request,"urlopen",side_effect=urllib.error.URLError("offline")) as urlopen,patch.object(tools.time,"sleep"):
   with self.assertRaises(tools.PubMedError):tools._get("esearch.fcgi",{"db":"pubmed"})
  self.assertEqual(urlopen.call_count,3)
if __name__=="__main__":unittest.main()
