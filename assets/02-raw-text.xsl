<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:tei="http://www.tei-c.org/ns/1.0"
    exclude-result-prefixes="tei"
    version="1.0">
    <xsl:output method="text"/>
    <xsl:template match="tei:TEI">
        <xsl:apply-templates select=".//tei:div" />
    </xsl:template>
    <xsl:template match="tei:div">
        <xsl:apply-templates select="tei:ab"/>
    </xsl:template>
    <xsl:template match="tei:note"/>
    <xsl:template match="tei:fw"/>
</xsl:stylesheet>