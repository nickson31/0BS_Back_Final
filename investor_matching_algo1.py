import re
from typing import List, Dict, Optional
from difflib import SequenceMatcher

def algorithm_1_enhanced_keyword_matching(
    user_profile: dict,
    investor_data: List[Dict],
    deep_research_keywords: Optional[List[str]] = None
) -> List[Dict]:
    """
    Algorithm 1: Primary Matching Logic using enhanced keyword and categorical matching.
    
    This algorithm focuses on direct and expanded keyword/categorical matching with
    strategic weighting for deep research keywords when available.
    
    Args:
        user_profile: Dictionary containing:
            - user_company_location: str
            - user_investment_categories: List[str]
            - user_investment_stage: str
        investor_data: List of investor dictionaries from Supabase
        deep_research_keywords: Optional list of 30 specialized keywords from Gemini
    
    Returns:
        List of investor dictionaries with added scoring fields, sorted by Final_Score
    """
    
    # Helper function to normalize text for matching
    def normalize_text(text: str) -> str:
        """Normalize text for better matching - lowercase and remove extra spaces"""
        return ' '.join(text.lower().strip().split())
    
    # Helper function to split comma-separated values
    def split_values(value: str) -> List[str]:
        """Split comma-separated string into normalized list"""
        if not value or not isinstance(value, str):
            return []
        return [normalize_text(v.strip()) for v in value.split(',') if v.strip()]
    
    # Helper function for fuzzy string matching
    def fuzzy_match_score(str1: str, str2: str, threshold: float = 0.7) -> float:
        """Calculate fuzzy match score between two strings"""
        ratio = SequenceMatcher(None, str1.lower(), str2.lower()).ratio()
        return ratio if ratio >= threshold else 0.0
    
    # Extract user profile data
    user_location = normalize_text(user_profile.get('user_company_location', ''))
    user_categories = [normalize_text(cat) for cat in user_profile.get('user_investment_categories', [])]
    user_stage = normalize_text(user_profile.get('user_investment_stage', ''))
    
    # Process deep research keywords if provided
    location_keywords = []
    stage_keywords = []
    category_keywords = []
    
    if deep_research_keywords:
        # Normalize deep research keywords
        deep_keywords_normalized = [normalize_text(kw) for kw in deep_research_keywords]
        
        # Heuristically categorize keywords (in production, this would be more sophisticated)
        # For now, we'll use simple pattern matching
        for keyword in deep_keywords_normalized:
            # Location indicators
            if any(loc_indicator in keyword for loc_indicator in ['city', 'country', 'region', 'area', 'usa', 'uk', 'india']):
                location_keywords.append(keyword)
            # Stage indicators
            elif any(stage_indicator in keyword for stage_indicator in ['seed', 'series', 'early', 'late', 'growth', 'venture']):
                stage_keywords.append(keyword)
            # Everything else is considered a category
            else:
                category_keywords.append(keyword)
    
    # Score each investor
    scored_investors = []
    
    for investor in investor_data:
        # Extract investor data
        investor_location = normalize_text(investor.get('Company_Location', ''))
        investor_categories = split_values(investor.get('Investment_Categories', ''))
        investor_stages = split_values(investor.get('Investment_Stage', ''))
        
        # 1. LOCATION MATCHING (Score_L1)
        location_score = 0.0
        
        # Direct location match
        if user_location and investor_location:
            # Check for exact match
            if user_location == investor_location:
                location_score = 1.0
            else:
                # Check if user location is contained in investor location or vice versa
                if user_location in investor_location or investor_location in user_location:
                    location_score = 0.8
                else:
                    # Fuzzy match for similar locations
                    location_score = fuzzy_match_score(user_location, investor_location, 0.6) * 0.7
        
        # Deep research keyword matching for location
        if location_keywords and investor_location:
            keyword_matches = sum(1 for kw in location_keywords if kw in investor_location)
            if keyword_matches > 0:
                # Boost score significantly for deep research matches
                keyword_boost = min(0.9, 0.3 * keyword_matches)
                location_score = max(location_score, keyword_boost)
        
        # Ensure score is normalized between 0 and 1
        location_score = min(1.0, max(0.0, location_score))
        
        # 2. INVESTMENT STAGE MATCHING (Score_S1)
        stage_score = 0.0
        
        if user_stage and investor_stages:
            # Direct stage match
            if user_stage in investor_stages:
                stage_score = 1.0
            else:
                # Check for partial matches (e.g., "seed" matches "pre-seed", "seed stage")
                for inv_stage in investor_stages:
                    if user_stage in inv_stage or inv_stage in user_stage:
                        stage_score = max(stage_score, 0.8)
                        break
                
                # Adjacent stage matching (simplified logic)
                stage_proximity = {
                    'pre-seed': ['seed', 'pre seed'],
                    'seed': ['pre-seed', 'pre seed', 'series a', 'early stage'],
                    'series a': ['seed', 'series b', 'early stage'],
                    'series b': ['series a', 'series c', 'growth'],
                    'early stage': ['seed', 'series a'],
                    'growth': ['series b', 'series c', 'late stage'],
                    'late stage': ['growth', 'series c', 'series d']
                }
                
                user_stage_key = user_stage.replace('-', ' ')
                for stage_key, adjacent_stages in stage_proximity.items():
                    if user_stage_key in stage_key or stage_key in user_stage_key:
                        for inv_stage in investor_stages:
                            if any(adj in inv_stage for adj in adjacent_stages):
                                stage_score = max(stage_score, 0.5)
        
        # Deep research keyword matching for stage
        if stage_keywords and investor_stages:
            for kw in stage_keywords:
                if any(kw in stage for stage in investor_stages):
                    stage_score = max(stage_score, 0.9)
        
        # Ensure score is normalized
        stage_score = min(1.0, max(0.0, stage_score))
        
        # 3. INVESTMENT CATEGORIES MATCHING (Score_C1)
        category_score = 0.0
        
        if user_categories and investor_categories:
            # Calculate overlap using Jaccard-like scoring
            exact_matches = 0
            partial_matches = 0
            
            for user_cat in user_categories:
                for inv_cat in investor_categories:
                    if user_cat == inv_cat:
                        exact_matches += 1
                    elif user_cat in inv_cat or inv_cat in user_cat:
                        partial_matches += 1
            
            # Calculate base score
            if exact_matches > 0:
                category_score = min(1.0, exact_matches / len(user_categories))
            elif partial_matches > 0:
                category_score = min(0.7, partial_matches / len(user_categories) * 0.7)
        
        # Deep research keyword matching for categories
        if category_keywords and investor_categories:
            keyword_matches = 0
            for kw in category_keywords:
                if any(kw in cat for cat in investor_categories):
                    keyword_matches += 1
            
            if keyword_matches > 0:
                # Significant boost for deep research matches
                keyword_boost = min(0.9, 0.2 * keyword_matches)
                category_score = max(category_score, keyword_boost)
        
        # Ensure score is normalized
        category_score = min(1.0, max(0.0, category_score))
        
        # Calculate final composite score (equal weighting: 33.3% each)
        final_score = (location_score * 0.333) + (stage_score * 0.333) + (category_score * 0.334)
        
        # Add scores to investor data
        scored_investor = investor.copy()
        scored_investor['Final_Score'] = round(final_score, 4)
        scored_investor['Location_Score'] = round(location_score, 4)
        scored_investor['Stage_Score'] = round(stage_score, 4)
        scored_investor['Category_Score'] = round(category_score, 4)
        
        scored_investors.append(scored_investor)
    
    # Sort by Final_Score in descending order
    scored_investors.sort(key=lambda x: x['Final_Score'], reverse=True)
    
    return scored_investors