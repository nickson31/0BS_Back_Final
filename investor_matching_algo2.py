import re
import math
from typing import List, Dict, Optional, Set
from collections import defaultdict

def algorithm_2_semantic_contextual_matching(
    user_profile: dict,
    investor_data: List[Dict],
    deep_research_keywords: Optional[List[str]] = None
) -> List[Dict]:
    """
    Algorithm 2: Semantic and Contextual Complement using broader matching logic.
    
    This algorithm uses semantic understanding and contextual analysis to find
    less obvious but potentially valuable investor matches.
    
    Args:
        user_profile: Dictionary containing:
            - user_company_location: str
            - user_investment_categories: List[str]
            - user_investment_stage: str
            - user_company_description: str (optional but recommended)
        investor_data: List of investor dictionaries from Supabase
        deep_research_keywords: Optional list of 30 specialized keywords from Gemini
    
    Returns:
        List of investor dictionaries with added scoring fields, sorted by Final_Score
    """
    
    # Helper function to calculate semantic similarity (placeholder)
    def calculate_semantic_similarity(text1: str, text2: str) -> float:
        """
        Placeholder for semantic similarity calculation.
        In production, this would use sentence transformers or similar.
        For now, using a simple keyword overlap approach.
        """
        if not text1 or not text2:
            return 0.0
        
        # Simple keyword extraction (in production, use proper NLP)
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words1 = words1 - stop_words
        words2 = words2 - stop_words
        
        if not words1 or not words2:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    # Define geographical regions for broader matching
    GEOGRAPHICAL_REGIONS = {
        'north_america': ['usa', 'united states', 'america', 'canada', 'mexico', 'north america'],
        'europe': ['europe', 'eu', 'uk', 'germany', 'france', 'spain', 'italy', 'netherlands', 'switzerland'],
        'asia': ['asia', 'china', 'japan', 'korea', 'singapore', 'india', 'hong kong', 'taiwan'],
        'middle_east': ['middle east', 'uae', 'dubai', 'saudi', 'israel', 'qatar', 'bahrain'],
        'latam': ['latin america', 'brazil', 'argentina', 'chile', 'colombia', 'mexico'],
        'africa': ['africa', 'nigeria', 'south africa', 'kenya', 'egypt'],
        'oceania': ['australia', 'new zealand', 'oceania']
    }
    
    # Define thematic category groups
    THEMATIC_GROUPS = {
        'tech_core': ['technology', 'software', 'saas', 'ai', 'ml', 'data', 'cloud', 'it'],
        'fintech': ['fintech', 'finance', 'banking', 'payments', 'lending', 'crypto', 'blockchain'],
        'health': ['health', 'healthcare', 'medical', 'biotech', 'pharma', 'wellness'],
        'sustainability': ['clean', 'green', 'sustainable', 'energy', 'climate', 'environmental'],
        'consumer': ['consumer', 'retail', 'ecommerce', 'marketplace', 'b2c', 'd2c'],
        'enterprise': ['enterprise', 'b2b', 'business', 'corporate', 'professional'],
        'deeptech': ['deep tech', 'robotics', 'quantum', 'space', 'advanced', 'frontier']
    }
    
    # Stage progression mapping for lenient matching
    STAGE_PROGRESSION = {
        'pre-seed': ['seed', 'angel', 'pre seed'],
        'seed': ['pre-seed', 'series a', 'angel', 'early stage'],
        'series a': ['seed', 'series b', 'early stage', 'venture'],
        'series b': ['series a', 'series c', 'growth', 'venture'],
        'series c': ['series b', 'series d', 'growth', 'late stage'],
        'growth': ['series b', 'series c', 'late stage', 'expansion'],
        'late stage': ['growth', 'series c', 'series d', 'pre-ipo']
    }
    
    # Helper function to find geographical region
    def find_region(location: str) -> Set[str]:
        """Find which geographical regions a location belongs to"""
        location_lower = location.lower()
        regions = set()
        for region, keywords in GEOGRAPHICAL_REGIONS.items():
            if any(kw in location_lower for kw in keywords):
                regions.add(region)
        return regions
    
    # Helper function to find thematic groups
    def find_themes(categories: List[str]) -> Set[str]:
        """Find which thematic groups the categories belong to"""
        themes = set()
        categories_text = ' '.join(categories).lower()
        for theme, keywords in THEMATIC_GROUPS.items():
            if any(kw in categories_text for kw in keywords):
                themes.add(theme)
        return themes
    
    # Extract user profile data
    user_location = user_profile.get('user_company_location', '').lower()
    user_categories = [cat.lower() for cat in user_profile.get('user_investment_categories', [])]
    user_stage = user_profile.get('user_investment_stage', '').lower()
    user_description = user_profile.get('user_company_description', '')
    
    # Find user's geographical regions and themes
    user_regions = find_region(user_location)
    user_themes = find_themes(user_categories)
    
    # Process deep research keywords
    dr_themes = set()
    dr_regions = set()
    dr_stage_indicators = set()
    
    if deep_research_keywords:
        dr_text = ' '.join(deep_research_keywords).lower()
        
        # Extract themes from deep research keywords
        for theme, keywords in THEMATIC_GROUPS.items():
            if any(kw in dr_text for kw in keywords):
                dr_themes.add(theme)
        
        # Extract regions from deep research keywords
        for region, keywords in GEOGRAPHICAL_REGIONS.items():
            if any(kw in dr_text for kw in keywords):
                dr_regions.add(region)
        
        # Extract stage indicators
        stage_keywords = ['seed', 'series', 'early', 'late', 'growth', 'venture', 'angel']
        for kw in stage_keywords:
            if kw in dr_text:
                dr_stage_indicators.add(kw)
    
    # Score each investor
    scored_investors = []
    
    for investor in investor_data:
        # Extract investor data
        investor_location = investor.get('Company_Location', '').lower()
        investor_categories = [cat.strip().lower() for cat in investor.get('Investment_Categories', '').split(',') if cat.strip()]
        investor_stages = [stage.strip().lower() for stage in investor.get('Investment_Stage', '').split(',') if stage.strip()]
        investor_description = investor.get('Company_Description', '')
        
        # Find investor's regions and themes
        investor_regions = find_region(investor_location)
        investor_themes = find_themes(investor_categories)
        
        # 1. SEMANTIC DESCRIPTION SIMILARITY (Score_D2)
        description_similarity = 0.0
        if user_description and investor_description:
            description_similarity = calculate_semantic_similarity(user_description, investor_description)
            
            # Boost if deep research keywords appear in both descriptions
            if deep_research_keywords:
                dr_boost = 0
                for kw in deep_research_keywords:
                    if kw.lower() in user_description.lower() and kw.lower() in investor_description.lower():
                        dr_boost += 0.05
                description_similarity = min(1.0, description_similarity + dr_boost)
        
        # 2. LOCATION MATCHING (Score_L2) - Regional approach
        location_score = 0.0
        
        # Direct location match
        if user_location and investor_location:
            if user_location in investor_location or investor_location in user_location:
                location_score = 0.9
            # Regional match
            elif user_regions and investor_regions:
                region_overlap = len(user_regions.intersection(investor_regions))
                if region_overlap > 0:
                    location_score = 0.7
            # Partial text match
            elif any(part in investor_location for part in user_location.split()):
                location_score = 0.5
        
        # Deep research regional boost
        if dr_regions and investor_regions:
            if dr_regions.intersection(investor_regions):
                location_score = max(location_score, 0.8)
        
        # Ensure normalized
        location_score = min(1.0, max(0.0, location_score))
        
        # 3. INVESTMENT STAGE MATCHING (Score_S2) - More lenient
        stage_score = 0.0
        
        if user_stage and investor_stages:
            # Direct match
            if any(user_stage in stage for stage in investor_stages):
                stage_score = 1.0
            else:
                # Check stage progression (adjacent stages)
                user_stage_key = user_stage.replace('-', ' ').strip()
                for base_stage, adjacent in STAGE_PROGRESSION.items():
                    if user_stage_key in base_stage:
                        for inv_stage in investor_stages:
                            if any(adj in inv_stage for adj in adjacent):
                                stage_score = max(stage_score, 0.7)
                
                # General venture capital matching
                if 'venture' in user_stage and any('venture' in s for s in investor_stages):
                    stage_score = max(stage_score, 0.6)
        
        # Deep research stage boost
        if dr_stage_indicators and investor_stages:
            for indicator in dr_stage_indicators:
                if any(indicator in stage for stage in investor_stages):
                    stage_score = max(stage_score, 0.8)
        
        # Ensure normalized
        stage_score = min(1.0, max(0.0, stage_score))
        
        # 4. INVESTMENT CATEGORIES MATCHING (Score_C2) - Thematic approach
        category_score = 0.0
        
        if user_categories and investor_categories:
            # Direct category overlap
            direct_matches = sum(1 for uc in user_categories if any(uc in ic for ic in investor_categories))
            if direct_matches > 0:
                category_score = min(1.0, direct_matches / len(user_categories))
            
            # Thematic overlap
            if user_themes and investor_themes:
                theme_overlap = len(user_themes.intersection(investor_themes))
                if theme_overlap > 0:
                    thematic_score = min(0.8, theme_overlap / len(user_themes) * 0.8)
                    category_score = max(category_score, thematic_score)
        
        # Deep research thematic boost
        if dr_themes and investor_themes:
            if dr_themes.intersection(investor_themes):
                category_score = max(category_score, 0.85)
        
        # Semantic category matching using description
        if category_score < 0.5 and user_description and investor_description:
            # Use description similarity as a category indicator
            category_score = max(category_score, description_similarity * 0.6)
        
        # Ensure normalized
        category_score = min(1.0, max(0.0, category_score))
        
        # Calculate final score with weighted components
        final_score = (
            description_similarity * 0.25 +  # Semantic understanding
            location_score * 0.20 +          # Regional presence
            stage_score * 0.25 +             # Investment stage alignment
            category_score * 0.30            # Thematic focus
        )
        
        # Add scores to investor data
        investor_with_scores = investor.copy()
        investor_with_scores.update({
            'Score_D2': round(description_similarity, 3),
            'Score_L2': round(location_score, 3),
            'Score_S2': round(stage_score, 3),
            'Score_C2': round(category_score, 3),
            'Final_Score': round(final_score, 3)
        })
        
        scored_investors.append(investor_with_scores)
    
    # Sort by final score descending
    scored_investors.sort(key=lambda x: x['Final_Score'], reverse=True)
    
    return scored_investors 