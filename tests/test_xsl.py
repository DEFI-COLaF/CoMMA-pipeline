"""Checks that assets/03-to-json.xsl emits exactly what
worker_convert_json.py's simplify_and_lines() reads: /doc @width|@height,
region @type|x|y|width|height, line @type|x|y|width|height + text.
"""
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

try:
    import lxml.etree as et
except ImportError:  # pragma: no cover
    et = None

ALTO = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Description>
    <sourceImageInformation><fileName>p0001.jpg</fileName></sourceImageInformation>
  </Description>
  <Tags>
    <OtherTag ID="BT1" LABEL="MainZone"/>
    <OtherTag ID="LT1" LABEL="DefaultLine"/>
  </Tags>
  <Layout>
    <Page WIDTH="1000" HEIGHT="1500" PHYSICAL_IMG_NR="1" ID="page1">
      <PrintSpace HPOS="0" VPOS="0" WIDTH="1000" HEIGHT="1500">
        <TextBlock ID="b1" TAGREFS="BT1" HPOS="100" VPOS="200" WIDTH="800" HEIGHT="1200">
          <TextLine ID="l1" TAGREFS="LT1" HPOS="110" VPOS="210" WIDTH="780" HEIGHT="40">
            <String CONTENT="In"/><SP/><String CONTENT="principio"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>"""


@unittest.skipIf(et is None, "lxml not installed")
class XslTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.xsl = et.XSLT(et.parse(str(REPO / "assets" / "03-to-json.xsl")))
        cls.doc = cls.xsl(et.fromstring(ALTO.encode()).getroottree())

    def test_doc_dimensions(self):
        doc = self.doc.xpath("/doc")[0]
        self.assertEqual(doc.attrib["width"], "1000")
        self.assertEqual(doc.attrib["height"], "1500")

    def test_region_geometry_and_type(self):
        region = self.doc.xpath("//region")[0]
        self.assertEqual(region.attrib["type"], "MainZone")
        self.assertEqual(
            (region.attrib["x"], region.attrib["y"],
             region.attrib["width"], region.attrib["height"]),
            ("100", "200", "800", "1200"),
        )

    def test_line_geometry_type_and_text(self):
        line = self.doc.xpath("//region/line")[0]
        self.assertEqual(line.attrib["type"], "DefaultLine")
        self.assertEqual(
            (line.attrib["x"], line.attrib["y"],
             line.attrib["width"], line.attrib["height"]),
            ("110", "210", "780", "40"),
        )
        self.assertEqual(str(line.text).strip(), "In principio")

    def test_attributes_present_even_when_source_lacks_them(self):
        # worker_convert_json.py does attrib["x"] (KeyError if absent) but
        # tolerates empty values via `or 0`: attributes must always exist.
        bare = ALTO.replace(' HPOS="110" VPOS="210" WIDTH="780" HEIGHT="40"', "")
        doc = self.xsl(et.fromstring(bare.encode()).getroottree())
        line = doc.xpath("//region/line")[0]
        for attr in ("x", "y", "width", "height"):
            self.assertIn(attr, line.attrib)
            self.assertEqual(line.attrib[attr], "")


if __name__ == "__main__":
    unittest.main()
