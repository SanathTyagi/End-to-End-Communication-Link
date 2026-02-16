from collections import Counter
import heapq

    
def get_huffman_bits(text):
    """
    Implements Huffman Source Coding (Entropy Encoding).
    
    Args:
        text (str): The input message string.
        
    Returns:
        encoded_output (list): Binary stream of compressed data.
        root (tuple): The root node of the Huffman Tree (required for decoding).
    """

    # 1. Frequency Analysis
    freq=Counter(text)
    
    # EDGE CASE: If text has only 1 unique char, Huffman tree cannot split.
    # We inject a dummy character '~' to force a tree structure (min 2 leaves)
    if(len(freq)==1):
        freq["~"]=1 
    # 2. Priority Queue Construction (Min-Heap)
    # Storing tuples as: (frequency, tie_breaker, node_data)    
    tie_breaker=0
    heap=[]
    for ch,f in freq.items():
        heapq.heappush(heap,(f,tie_breaker,ch))
        tie_breaker+=1


    # 3. Build Huffman Tree (Bottom-Up Approach)
    # Continuously merge the two nodes with the lowest frequency
    while len(heap)>1:
        f1,tiebreaker,d1=heapq.heappop(heap) # Smallest frequency
        f2,tiebreaker,d2=heapq.heappop(heap) # Second smallest frequency

        # Push combined parent node back to heap
        # Parent frequency = sum of children. Tie_breaker ensures stability.
        heapq.heappush(heap,(f1+f2,tie_breaker,(d1,d2)))
        tie_breaker+=1
    # Extract the final root of the tree
    root=heapq.heappop(heap)[2]
    # 4. Generate Codebook
    huffman={} # Maps char -> bit_list (e.g., 'a': [0, 1])

    
    def generateCode(node ,path):
        """
        Traverses tree to assign bits: 0 for Left, 1 for Right.
        """
        # Base Case: Leaf Node (Character found)
        if(isinstance(node,str)):
            huffman[node]=path[:]
            return

        # Recursive Step: Internal Node (Tuple of children)
        left_child,right_child=node
        
        # Traverse Left
        path.append(0)
        generateCode(left_child,path)
        path.pop()
        # Traverse Right 
        path.append(1)
        generateCode(right_child,path)
        path.pop()


    # Start recursion
    lst=[]
    generateCode(root,lst)

    # 5. Encode Stream
    encoded_output=[]
    for char in text:
        code_string=huffman[char]
        for bit in code_string:
            encoded_output.append(int(bit))
    return encoded_output,root


#decoding 
def decode_huffman_bits(decoded_bit,root):
    """
    Decodes the compressed bitstream using the Huffman Tree.
    
    Args:
        decoded_bit (list): The compressed binary stream.
        root (tuple): The Huffman Tree structure returned by the encoder.
        
    Returns:
        message (str): The reconstructed original text.
    """

    
    message=""
    node=root
    j=0
    # Traverse the bitstream
    while(j<len(decoded_bit)):
        # Unpack the current node into children
        # NOTE: 'node' is guaranteed to be a tuple here because we reset 
        # to 'root' immediately after finding a leaf (string).
        left_child,right_child=node
        if(decoded_bit[j]==0):
            # Move Left
            node=left_child
            if(isinstance(node,str)):
                message+=node
                node=root # Reset to top of tree for next character
            j+=1
        else:
            # Move Right
            node=right_child
            if(isinstance(node,str)):
                message+=node
                node=root # Reset to top of tree for next character
            j+=1

    return message
        

