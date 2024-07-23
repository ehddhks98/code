import random

def create_episode(data, classes, n_way, n_support, n_query):
    # n_way 개의 클래스를 랜덤으로 선택
    selected_classes = random.sample(classes, n_way)
    
    support_indices = []  # 지원 샘플 인덱스를 저장할 리스트
    query_indices = []    # 쿼리 샘플 인덱스를 저장할 리스트
    
    for cls in selected_classes:
        # 현재 클래스에 해당하는 모든 데이터 인덱스를 가져옴
        cls_indices = [i for i, t in enumerate(data.targets) if t == cls]
        
        # 지원 샘플로 사용할 n_support개의 인덱스를 랜덤으로 선택
        support = random.sample(cls_indices, n_support)
        
        # 나머지 샘플들 중에서 쿼리 샘플로 사용할 n_query개의 인덱스를 랜덤으로 선택
        query = random.sample([i for i in cls_indices if i not in support], n_query)
        
        # 지원 샘플과 쿼리 샘플 인덱스를 각각 리스트에 추가
        support_indices.extend(support)
        query_indices.extend(query)
    
    return support_indices, query_indices  # 지원 샘플 인덱스와 쿼리 샘플 인덱스를 반환