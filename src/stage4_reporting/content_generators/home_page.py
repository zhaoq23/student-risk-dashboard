"""
home_page.py
"""
import datetime

import datetime

    
def generate_home_page():
    """Create home """
    current_date = datetime.datetime.now().strftime("%B %d %Y")
    
    return f"""
    <div class="card">
        <h2>Project Overview</h2>
        
        <!-- key card -->
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0;">
            
            <!-- Card 1: Total Students -->
            <div style="background: linear-gradient(135deg, #1E40AF 0%, #1E3A8A 100%); 
                        padding: 24px; border-radius: 12px; text-align: center; color: white;">
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">Total Students Analyzed</div>
                <div style="font-size: 42px; font-weight: 700;">4,274</div>
            </div>
            
            <!-- Card 2: Dataset Dropout Rate -->
            <div style="background: linear-gradient(135deg, #A855F7 0%, #9333EA 100%); 
                        padding: 24px; border-radius: 12px; text-align: center; color: white;">
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">Dataset Dropout Rate</div>
                <div style="font-size: 42px; font-weight: 700;">32.9%</div>
            </div>
            
            <!-- Card 3: U.S. National Rate -->
            <div style="background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%); 
                        padding: 24px; border-radius: 12px; text-align: center; color: white;">
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">U.S. National Dropout Rate</div>
                <div style="font-size: 42px; font-weight: 700;">39%</div>
            </div>
            
            <!-- Card 4: NY Growth Rate -->
            <div style="background: linear-gradient(135deg, #5B9FED 0%, #4F8FDB 100%); 
                        padding: 24px; border-radius: 12px; text-align: center; color: white;">
                <div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">NY Non-Completers Growth</div>
                <div style="font-size: 42px; font-weight: 700;">+2.3%</div>
                <div style="font-size: 11px; opacity: 0.85; margin-top: 4px;">1.9M individuals</div>
            </div>
            
        </div>
        
        <!-- Background Section -->
        <div style="margin: 30px 0;">
            <h3 style="color: #1e293b; margin-bottom: 16px;">Background</h3>
            <p style="line-height: 1.8; color: #475569; font-size: 15px;">
                While high school dropout rates are declining, a staggering 
                <strong>32.9%</strong><sup><a href="https://research.com/universities-colleges/college-dropout-rates#:~:text=In%20a%20nutshell%2C%20failing%20to,school%20without%20completing%20their%20degree." 
                                            target="_blank" 
                                            style="color: #3b82f6; text-decoration: none; font-size: 12px;">[1]</a></sup> 
                of U.S. college students still fail to graduate. The core priority for education leaders lies in bridging 
                these gaps by identifying the <strong>one-in-four students</strong> currently at risk of dropping out 
                before they lose momentum.
            </p>
            <p style="line-height: 1.8; color: #475569; font-size: 15px; margin-top: 12px;">
                In New York alone, there are over <strong>1.9 million individuals</strong> with some college but no credential, 
                a population that grew by 
                <strong>2.3%</strong><sup><a href="https://educationdata.org/college-dropout-rates#ny" 
                                            target="_blank" 
                                            style="color: #3b82f6; text-decoration: none; font-size: 12px;">[2]</a></sup> 
                in the past year.
            </p>
        </div>
        
        <div class="divider"></div>
        
        <!-- Mission Section -->
        <div style="margin: 30px 0;">
            <h3 style="color: #1e293b; margin-bottom: 16px;">Our Mission</h3>
            <p style="line-height: 1.8; color: #475569; font-size: 15px;">
                This app aims to <strong>reduce academic failure</strong> by using machine learning to identify at-risk students 
                at an early stage. By providing actionable insights, we enable educators to implement timely support strategies 
                that keep students on the path to success.
            </p>
        </div>
        
        <div class="divider"></div>
        
        <!-- Framework Section -->
        <div style="margin: 30px 0;">
            <h3 style="color: #1e293b; margin-bottom: 16px;">Framework & Solutions</h3>
            <p style="line-height: 1.8; color: #475569; font-size: 15px; margin-bottom: 20px;">
                We provide a systematic approach to student success across three dimensions:
            </p>

            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                
                <!-- Student Level -->
                <a onclick="showSection('student')" 
                style="background: #f1f5f9; padding: 24px; border-radius: 8px; 
                        border-left: 4px solid #4169E1; text-decoration: none; 
                        cursor: pointer; transition: all 0.3s ease;
                        display: block;"
                onmouseover="this.style.background='#e2e8f0'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(65, 105, 225, 0.15)';"
                onmouseout="this.style.background='#f1f5f9'; this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                    <h4 style="color: #1e293b; margin: 0 0 8px 0; font-size: 18px;">Student-Level</h4>
                    <p style="margin: 0; color: #64748b; font-size: 14px; line-height: 1.6;">
                        Individual risk profiles for personalized intervention
                    </p>
                    <div style="margin-top: 12px; color: #4169E1; font-size: 13px; font-weight: 500;">
                        View Details →
                    </div>
                </a>
                
                <!-- Cohort Level -->
                <a onclick="showSection('cohort')" 
                style="background: #f1f5f9; padding: 24px; border-radius: 8px; 
                        border-left: 4px solid #9333EA; text-decoration: none; 
                        cursor: pointer; transition: all 0.3s ease;
                        display: block;"
                onmouseover="this.style.background='#e2e8f0'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(147, 51, 234, 0.15)';"
                onmouseout="this.style.background='#f1f5f9'; this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                    <h4 style="color: #1e293b; margin: 0 0 8px 0; font-size: 18px;">Cohort-Level</h4>
                    <p style="margin: 0; color: #64748b; font-size: 14px; line-height: 1.6;">
                        Trend analysis for specific student demographics
                    </p>
                    <div style="margin-top: 12px; color: #9333EA; font-size: 13px; font-weight: 500;">
                        View Details →
                    </div>
                </a>
                
                <!-- System Level -->
                <a onclick="showSection('system')" 
                style="background: #f1f5f9; padding: 24px; border-radius: 8px; 
                        border-left: 4px solid #DC2626; text-decoration: none; 
                        cursor: pointer; transition: all 0.3s ease;
                        display: block;"
                onmouseover="this.style.background='#e2e8f0'; this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(220, 38, 38, 0.15)';"
                onmouseout="this.style.background='#f1f5f9'; this.style.transform='translateY(0)'; this.style.boxShadow='none';">
                    <h4 style="color: #1e293b; margin: 0 0 8px 0; font-size: 18px;">System-Level</h4>
                    <p style="margin: 0; color: #64748b; font-size: 14px; line-height: 1.6;">
                        Strategic dashboards for school and district leadership
                    </p>
                    <div style="margin-top: 12px; color: #DC2626; font-size: 13px; font-weight: 500;">
                        View Details →
                    </div>
                </a>
                
            </div>
        </div>
        
        <div class="divider"></div>
        
        <!-- Data Reference Section -->
        <div style="margin: 30px 0;">
            <h3 style="color: #1e293b; margin-bottom: 16px;">Data Reference</h3>
            <p style="line-height: 1.8; color: #475569; font-size: 15px;">
                This project utilizes a dataset from the 
                <a href="https://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success" 
                   target="_blank" 
                   style="color: #3b82f6; text-decoration: underline;">
                    UCI Machine Learning Repository
                </a> 
                to present these real-world challenges. Originally created to help reduce academic attrition in higher education, 
                this data allows us to demonstrate how machine learning can effectively flag students at risk during their 
                academic journey.
            </p>
        </div>
        
        <div class="divider"></div>
        
        <!-- Footer: Created By -->
        <div style="margin: 30px 0; padding: 20px; background: #f8fafc; border-radius: 8px; text-align: center;">
            <p style="margin: 0; color: #64748b; font-size: 14px; line-height: 1.8;">
                <strong style="color: #1e293b;">Created by:</strong> Qi Zhao 
                <span style="color: #94a3b8;">•</span>
                <a href="mailto:zhaoq23009@gmail.com" 
                   style="color: #3b82f6; text-decoration: none;">
                    zhaoq23009@gmail.com
                </a>
            </p>
            <p style="margin: 8px 0 0 0; color: #94a3b8; font-size: 13px;">
                Last updated: {current_date}
            </p>
        </div>
        
    </div>
    """